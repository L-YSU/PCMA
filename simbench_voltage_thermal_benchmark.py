from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandapower as pp
import simbench as sb

from optimize_ai_storage_dispatch import solve_storage_dispatch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT / "datasets"
OUT = PACKAGE_ROOT / "results" / "00_analysis_outputs"
FIG = OUT / "figures"

DEFAULT_NETWORKS = [
    "1-HVMV-mixed-1.105-0-no_sw",
    "1-HVMV-mixed-2.102-0-no_sw",
    "1-HVMV-mixed-4.101-0-no_sw",
    "1-HVMV-mixed-1.105-1-no_sw",
]


def representative_hours(label: str) -> list[int]:
    if label == "8h":
        return [0, 3, 6, 9, 12, 15, 18, 21]
    if label == "12h":
        return list(range(0, 24, 2))
    if label == "24h":
        return list(range(24))
    raise ValueError(f"Unknown hours label: {label}")


def load_scenario(hours: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    scenario = pd.read_csv(OUT / "ieee33_ai_datacenter_24h_scenario_moderate.csv")
    scenario = scenario[scenario["hour"].isin(hours)].reset_index(drop=True)
    raw = scenario["ai_load_mw"].to_numpy(dtype=float)
    raw_norm = raw / max(float(raw.max()), 1e-9)
    price = scenario.get("price_pu", pd.Series(np.ones(len(scenario)))).to_numpy(dtype=float)
    background = scenario.get(
        "background_load_multiplier", pd.Series(np.ones(len(scenario)))
    ).to_numpy(dtype=float)
    return raw_norm, price, background, scenario


def prepare_net(code: str):
    net = sb.get_simbench_net(code)
    if hasattr(net, "controller"):
        net.controller.drop(net.controller.index, inplace=True)
    return net


def run_power_flow(net) -> None:
    pp.runpp(
        net,
        algorithm="nr",
        numba=False,
        max_iteration=100,
        calculate_voltage_angles=False,
        check_connectivity=True,
    )


def choose_ai_bus(net) -> int:
    probe = copy.deepcopy(net)
    run_power_flow(probe)
    load_buses = set(probe.load.bus.astype(int).tolist()) if len(probe.load) else set(probe.bus.index)
    res = probe.res_bus.copy()
    res = res[res.index.isin(load_buses)]
    if "vn_kv" in probe.bus.columns:
        candidate_index = probe.bus[probe.bus.vn_kv <= probe.bus.vn_kv.quantile(0.75)].index
        filtered = res[res.index.isin(candidate_index)]
        if len(filtered):
            res = filtered
    return int(res.vm_pu.idxmin())


def scale_ai_profile(net, raw_norm: np.ndarray, fraction_of_base_load: float = 0.015) -> np.ndarray:
    base_p = float(net.load.p_mw.sum()) if len(net.load) else 1.0
    peak = max(0.05, fraction_of_base_load * base_p)
    return peak * raw_norm


def verify_profile(
    code: str,
    base_net,
    ai_bus: int,
    ai_profile: np.ndarray,
    background: np.ndarray,
    mode: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    original_load_p = base_net.load.p_mw.copy() if len(base_net.load) else None
    original_load_q = base_net.load.q_mvar.copy() if len(base_net.load) else None

    for step, ai_mw in enumerate(ai_profile):
        net = copy.deepcopy(base_net)
        if len(net.load):
            net.load.loc[:, "p_mw"] = original_load_p.values * float(background[step])
            net.load.loc[:, "q_mvar"] = original_load_q.values * float(background[step])
        if ai_mw > 1e-9:
            pp.create_load(
                net,
                ai_bus,
                p_mw=float(ai_mw),
                q_mvar=0.25 * float(ai_mw),
                name=f"ai_datacenter_{mode}",
            )
        try:
            run_power_flow(net)
            records.append(
                {
                    "code": code,
                    "ai_bus": ai_bus,
                    "mode": mode,
                    "step": step,
                    "converged": bool(net.converged),
                    "ai_load_grid_seen_mw": float(ai_mw),
                    "min_voltage_pu": float(net.res_bus.vm_pu.min()),
                    "max_voltage_pu": float(net.res_bus.vm_pu.max()),
                    "max_line_loading_percent": float(net.res_line.loading_percent.max()) if len(net.line) else 0.0,
                    "overloaded_line_count": int((net.res_line.loading_percent > 100.0).sum()) if len(net.line) else 0,
                    "undervoltage_bus_count_lt_0p95": int((net.res_bus.vm_pu < 0.95).sum()),
                    "overvoltage_bus_count_gt_1p05": int((net.res_bus.vm_pu > 1.05).sum()),
                    "line_losses_mw": float(net.res_line.pl_mw.sum()) if len(net.line) else 0.0,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "code": code,
                    "ai_bus": ai_bus,
                    "mode": mode,
                    "step": step,
                    "converged": False,
                    "ai_load_grid_seen_mw": float(ai_mw),
                    "error": repr(exc),
                }
            )
    return pd.DataFrame(records)


def build_voltage_stress_and_upper(no_ai: pd.DataFrame, raw_ai: pd.DataFrame, raw_profile: np.ndarray, background: np.ndarray):
    no_ai = no_ai.sort_values("step")
    raw_ai = raw_ai.sort_values("step")
    base_v = no_ai["min_voltage_pu"].to_numpy(dtype=float)
    raw_v = raw_ai["min_voltage_pu"].to_numpy(dtype=float)
    uv = raw_ai["undervoltage_bus_count_lt_0p95"].to_numpy(dtype=float)
    ov = raw_ai["overvoltage_bus_count_gt_1p05"].to_numpy(dtype=float)

    gap = np.maximum(0.0, 0.95 - raw_v)
    stress = 1.0 + 30.0 * gap + 0.04 * uv + 0.02 * ov + 0.4 * background
    stress = np.nan_to_num(stress, nan=10.0, posinf=10.0, neginf=1.0)

    slope = (base_v - raw_v) / np.maximum(raw_profile, 1e-6)
    upper = np.full(len(raw_profile), float(raw_profile.max()))
    mask = slope > 1e-6
    upper[mask] = (base_v[mask] - 0.952) / slope[mask]
    upper = np.nan_to_num(upper, nan=float(raw_profile.max()), posinf=float(raw_profile.max()), neginf=0.0)
    upper = np.clip(upper, 0.0, float(raw_profile.max()))
    return stress, upper


def add_thermal_repair(stress: np.ndarray, upper: np.ndarray, raw_ai_result: pd.DataFrame, raw_profile: np.ndarray):
    raw_ai_result = raw_ai_result.sort_values("step")
    max_loading = raw_ai_result["max_line_loading_percent"].to_numpy(dtype=float)
    overload_count = raw_ai_result["overloaded_line_count"].to_numpy(dtype=float)
    thermal_excess = np.maximum(0.0, max_loading - 100.0) / 100.0
    thermal_stress = 1.0 + 4.0 * thermal_excess + 0.08 * overload_count
    new_stress = np.asarray(stress, dtype=float) * thermal_stress

    new_upper = np.asarray(upper, dtype=float).copy()
    overloaded = max_loading > 100.0
    safe_fraction = np.ones_like(max_loading, dtype=float)
    safe_fraction[overloaded] = np.clip(98.0 / max_loading[overloaded], 0.0, 1.0)
    new_upper[overloaded] = np.minimum(new_upper[overloaded], raw_profile[overloaded] * safe_fraction[overloaded])
    return new_stress, np.clip(new_upper, 0.0, float(np.max(raw_profile)))


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["code", "ai_bus", "mode"])
        .agg(
            converged_hours=("converged", "sum"),
            min_voltage_min=("min_voltage_pu", "min"),
            min_voltage_mean=("min_voltage_pu", "mean"),
            max_voltage_max=("max_voltage_pu", "max"),
            max_line_loading_max=("max_line_loading_percent", "max"),
            overloaded_line_hours=("overloaded_line_count", "sum"),
            undervoltage_bus_hours=("undervoltage_bus_count_lt_0p95", "sum"),
            overvoltage_bus_hours=("overvoltage_bus_count_gt_1p05", "sum"),
            losses_mw_mean=("line_losses_mw", "mean"),
            ai_grid_seen_mw_peak=("ai_load_grid_seen_mw", "max"),
        )
        .reset_index()
    )


def run_benchmark(network_codes: list[str], hours_label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hours = representative_hours(hours_label)
    raw_norm, price, background, _scenario = load_scenario(hours)
    all_results: list[pd.DataFrame] = []
    dispatches: list[pd.DataFrame] = []
    network_records: list[dict[str, object]] = []

    for code in network_codes:
        base_net = prepare_net(code)
        ai_bus = choose_ai_bus(base_net)
        raw_ai = scale_ai_profile(base_net, raw_norm)

        no_ai = verify_profile(code, base_net, ai_bus, np.zeros_like(raw_ai), background, "no_ai")
        raw = verify_profile(code, base_net, ai_bus, raw_ai, background, "raw_ai")

        v_stress, v_upper = build_voltage_stress_and_upper(no_ai, raw, raw_ai, background)
        voltage_dispatch = solve_storage_dispatch(
            raw_ai,
            v_stress,
            price,
            grid_upper_mw=v_upper,
            battery_power_mw=max(0.05, 0.50 * float(raw_ai.max())),
            battery_capacity_mwh=max(0.2, 2.5 * float(raw_ai.max())),
            initial_soc_mwh=max(0.1, 1.25 * float(raw_ai.max())),
        )
        voltage_dispatch["code"] = code
        voltage_dispatch["ai_bus"] = ai_bus
        voltage_dispatch["strategy"] = "voltage_only_storage"
        voltage = verify_profile(
            code,
            base_net,
            ai_bus,
            voltage_dispatch["optimized_grid_ai_mw"].to_numpy(dtype=float),
            background,
            "voltage_only_storage",
        )

        t_stress, t_upper = add_thermal_repair(v_stress, v_upper, raw, raw_ai)
        thermal_dispatch = solve_storage_dispatch(
            raw_ai,
            t_stress,
            price,
            grid_upper_mw=t_upper,
            battery_power_mw=max(0.05, 0.50 * float(raw_ai.max())),
            battery_capacity_mwh=max(0.2, 2.5 * float(raw_ai.max())),
            initial_soc_mwh=max(0.1, 1.25 * float(raw_ai.max())),
        )
        thermal_dispatch["code"] = code
        thermal_dispatch["ai_bus"] = ai_bus
        thermal_dispatch["strategy"] = "voltage_thermal_storage"
        thermal = verify_profile(
            code,
            base_net,
            ai_bus,
            thermal_dispatch["optimized_grid_ai_mw"].to_numpy(dtype=float),
            background,
            "voltage_thermal_storage",
        )

        all_results.extend([no_ai, raw, voltage, thermal])
        dispatches.extend([voltage_dispatch, thermal_dispatch])
        network_records.append(
            {
                "code": code,
                "ai_bus": ai_bus,
                "bus_count": len(base_net.bus),
                "line_count": len(base_net.line),
                "load_count": len(base_net.load),
                "sgen_count": len(base_net.sgen),
                "base_load_mw": float(base_net.load.p_mw.sum()) if len(base_net.load) else 0.0,
                "ai_peak_mw": float(raw_ai.max()),
            }
        )

    results = pd.concat(all_results, ignore_index=True)
    dispatch = pd.concat(dispatches, ignore_index=True)
    networks = pd.DataFrame(network_records)
    summary = summarize(results)
    return networks, dispatch, results, summary


def write_outputs(prefix: str, networks: pd.DataFrame, dispatch: pd.DataFrame, results: pd.DataFrame, summary: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    networks.to_csv(OUT / f"{prefix}_networks.csv", index=False, encoding="utf-8-sig")
    dispatch.to_csv(OUT / f"{prefix}_dispatch.csv", index=False, encoding="utf-8-sig")
    results.to_csv(OUT / f"{prefix}_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")

    pivot = summary.pivot_table(
        index="code",
        columns="mode",
        values=["undervoltage_bus_hours", "overloaded_line_hours", "min_voltage_min", "max_line_loading_max"],
        aggfunc="first",
    )
    pivot.to_csv(OUT / f"{prefix}_comparison_pivot.csv", encoding="utf-8-sig")

    plt.figure(figsize=(11, 4))
    modes = ["raw_ai", "voltage_only_storage", "voltage_thermal_storage"]
    x = np.arange(len(summary["code"].unique()))
    width = 0.24
    for i, mode in enumerate(modes):
        values = []
        for code in summary["code"].unique():
            row = summary[summary["code"].eq(code) & summary["mode"].eq(mode)]
            values.append(float(row["overloaded_line_hours"].iloc[0]) if len(row) else 0.0)
        plt.bar(x + (i - 1) * width, values, width=width, label=mode)
    plt.xticks(x, summary["code"].unique(), rotation=18, ha="right")
    plt.ylabel("Overloaded line-hours")
    plt.title(prefix.replace("_", " "))
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / f"{prefix}_overload_comparison.png", dpi=200)
    plt.close()

    log = f"""# {prefix} Benchmark Log

## Summary

```csv
{summary.to_csv(index=False)}```

## Interpretation Guide

Use `raw_ai` to quantify the unmitigated AI data-center load impact.
Use `voltage_only_storage` as the earlier storage dispatch baseline.
Use `voltage_thermal_storage` as the proposed stronger repair strategy that
adds line-loading-aware repair signals.
"""
    (OUT / f"{prefix.upper()}_LOG.md").write_text(log, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="SimBench voltage-only vs voltage+thermal benchmark.")
    parser.add_argument("--hours", choices=["8h", "12h", "24h"], default="12h")
    parser.add_argument("--networks", nargs="*", default=DEFAULT_NETWORKS)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    prefix = args.prefix or f"simbench_voltage_thermal_{args.hours}_{len(args.networks)}net"
    networks, dispatch, results, summary = run_benchmark(args.networks, args.hours)
    write_outputs(prefix, networks, dispatch, results, summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
