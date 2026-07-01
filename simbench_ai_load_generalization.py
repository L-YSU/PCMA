from pathlib import Path

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


CANDIDATE_CODES = [
    "1-HVMV-mixed-1.105-0-no_sw",
    "1-HVMV-mixed-2.102-0-no_sw",
]

REPRESENTATIVE_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]


def load_scenario():
    scenario = pd.read_csv(OUT / "ieee33_ai_datacenter_24h_scenario_moderate.csv")
    scenario = scenario[scenario["hour"].isin(REPRESENTATIVE_HOURS)].reset_index(drop=True)
    raw_shape = scenario["ai_load_mw"].to_numpy(dtype=float)
    raw_norm = raw_shape / max(raw_shape.max(), 1e-9)
    price = scenario.get("price_pu", pd.Series(np.ones(len(scenario)))).to_numpy(dtype=float)
    background = scenario.get(
        "background_load_multiplier", pd.Series(np.ones(len(scenario)))
    ).to_numpy(dtype=float)
    return raw_norm, price, background


def prepare_net(code):
    net = sb.get_simbench_net(code)
    # Disable controllers/time-series apparatus for static power-flow screening.
    if hasattr(net, "controller"):
        net.controller.drop(net.controller.index, inplace=True)
    return net


def choose_ai_bus(net):
    """Choose a stressed but valid load bus after baseline power flow."""
    pp.runpp(
        net,
        algorithm="nr",
        numba=False,
        max_iteration=100,
        calculate_voltage_angles=False,
        check_connectivity=True,
    )
    load_buses = set(net.load.bus.astype(int).tolist()) if len(net.load) else set(net.bus.index)
    res = net.res_bus.copy()
    res = res[res.index.isin(load_buses)]
    # Avoid extremely high-voltage buses when possible.
    if "vn_kv" in net.bus.columns:
        candidate_index = net.bus[net.bus.vn_kv <= net.bus.vn_kv.quantile(0.75)].index
        filtered = res[res.index.isin(candidate_index)]
        if len(filtered):
            res = filtered
    return int(res.vm_pu.idxmin())


def scale_ai_profile(net, raw_norm, fraction_of_base_load=0.08):
    base_p = float(net.load.p_mw.sum()) if len(net.load) else 1.0
    peak = max(0.05, fraction_of_base_load * base_p)
    return peak * raw_norm


def run_timeseries_verification(code, ai_bus, ai_profile, background, mode):
    records = []
    base_net = prepare_net(code)
    original_load_p = base_net.load.p_mw.copy() if len(base_net.load) else None
    original_load_q = base_net.load.q_mvar.copy() if len(base_net.load) else None

    for hour, ai_mw in enumerate(ai_profile):
        net = prepare_net(code)
        if len(net.load):
            net.load.loc[:, "p_mw"] = original_load_p.values * float(background[hour])
            net.load.loc[:, "q_mvar"] = original_load_q.values * float(background[hour])
        if ai_mw > 1e-9:
            pp.create_load(
                net,
                ai_bus,
                p_mw=float(ai_mw),
                q_mvar=0.25 * float(ai_mw),
                name=f"ai_datacenter_{mode}",
            )
        try:
            pp.runpp(
                net,
                algorithm="nr",
                numba=False,
                max_iteration=100,
                calculate_voltage_angles=False,
                check_connectivity=True,
            )
            records.append(
                {
                    "code": code,
                    "ai_bus": ai_bus,
                    "mode": mode,
                    "hour": hour,
                    "converged": bool(net.converged),
                    "ai_load_grid_seen_mw": float(ai_mw),
                    "min_voltage_pu": float(net.res_bus.vm_pu.min()),
                    "max_voltage_pu": float(net.res_bus.vm_pu.max()),
                    "max_line_loading_percent": float(net.res_line.loading_percent.max())
                    if len(net.line)
                    else 0.0,
                    "overloaded_line_count": int((net.res_line.loading_percent > 100).sum())
                    if len(net.line)
                    else 0,
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
                    "hour": hour,
                    "converged": False,
                    "ai_load_grid_seen_mw": float(ai_mw),
                    "error": repr(exc),
                }
            )
    return pd.DataFrame(records)


def build_stress_and_upper(no_ai, raw_ai, raw_profile, background):
    no_ai = no_ai.sort_values("hour")
    raw_ai = raw_ai.sort_values("hour")
    raw_profile = np.asarray(raw_profile, dtype=float)
    base_v = no_ai["min_voltage_pu"].to_numpy(dtype=float)
    raw_v = raw_ai["min_voltage_pu"].to_numpy(dtype=float)
    base_v = np.nan_to_num(base_v, nan=0.90, posinf=1.10, neginf=0.80)
    raw_v = np.nan_to_num(raw_v, nan=0.85, posinf=1.10, neginf=0.80)
    uv = raw_ai["undervoltage_bus_count_lt_0p95"].to_numpy(dtype=float)
    ov = raw_ai["overvoltage_bus_count_gt_1p05"].to_numpy(dtype=float)
    uv = np.nan_to_num(uv, nan=10.0, posinf=10.0, neginf=0.0)
    ov = np.nan_to_num(ov, nan=10.0, posinf=10.0, neginf=0.0)
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


def summarize(results):
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


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    raw_norm, price, background = load_scenario()
    all_results = []
    dispatches = []
    network_records = []

    for code in CANDIDATE_CODES:
        net = prepare_net(code)
        ai_bus = choose_ai_bus(net)
        raw_ai = scale_ai_profile(net, raw_norm, fraction_of_base_load=0.015)
        no_ai = run_timeseries_verification(
            code, ai_bus, np.zeros_like(raw_ai), background, "no_ai"
        )
        raw = run_timeseries_verification(code, ai_bus, raw_ai, background, "raw_ai")
        stress, upper = build_stress_and_upper(no_ai, raw, raw_ai, background)
        dispatch = solve_storage_dispatch(
            raw_ai,
            stress,
            price,
            grid_upper_mw=upper,
            battery_power_mw=max(0.05, 0.50 * float(raw_ai.max())),
            battery_capacity_mwh=max(0.2, 2.5 * float(raw_ai.max())),
            initial_soc_mwh=max(0.1, 1.25 * float(raw_ai.max())),
        )
        dispatch["code"] = code
        dispatch["ai_bus"] = ai_bus
        dispatches.append(dispatch)
        opt = run_timeseries_verification(
            code,
            ai_bus,
            dispatch["optimized_grid_ai_mw"].to_numpy(dtype=float),
            background,
            "optimized_storage",
        )
        all_results.extend([no_ai, raw, opt])
        network_records.append(
            {
                "code": code,
                "ai_bus": ai_bus,
                "bus_count": len(net.bus),
                "line_count": len(net.line),
                "load_count": len(net.load),
                "sgen_count": len(net.sgen),
                "base_load_mw": float(net.load.p_mw.sum()) if len(net.load) else 0.0,
                "ai_peak_mw": float(raw_ai.max()),
            }
        )

    results = pd.concat(all_results, ignore_index=True)
    dispatch = pd.concat(dispatches, ignore_index=True)
    summary = summarize(results)
    pd.DataFrame(network_records).to_csv(
        OUT / "simbench_ai_generalization_networks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    dispatch.to_csv(
        OUT / "simbench_ai_generalization_dispatch.csv",
        index=False,
        encoding="utf-8-sig",
    )
    results.to_csv(
        OUT / "simbench_ai_generalization_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        OUT / "simbench_ai_generalization_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(10, 4))
    for code in CANDIDATE_CODES:
        data = summary[summary["code"].eq(code)].set_index("mode")
        if "raw_ai" in data.index and "optimized_storage" in data.index:
            plt.plot(
                ["raw_ai", "optimized_storage"],
                [
                    data.loc["raw_ai", "undervoltage_bus_hours"],
                    data.loc["optimized_storage", "undervoltage_bus_hours"],
                ],
                marker="o",
                label=code.replace("1-HVMV-mixed-", ""),
            )
    plt.ylabel("Undervoltage bus-hours")
    plt.title("SimBench Generalization: Raw AI vs Optimized Storage")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "simbench_generalization_undervoltage.png", dpi=200)
    plt.close()

    (OUT / "SIMBENCH_GENERALIZATION_LOG.md").write_text(
        f"""# SimBench Generalization Log

## Purpose

Verify that the AI-load dispatch idea is not limited to IEEE 33-bus.

## Networks

```csv
{pd.DataFrame(network_records).to_csv(index=False)}```

## Summary

```csv
{summary.to_csv(index=False)}```

## Interpretation

The selected SimBench networks are larger and more realistic than IEEE 33-bus.
This experiment keeps the setup deliberately simple: one AI data-center load is
placed at a stressed load bus, raw AI and optimized-storage schedules are
verified by pandapower, and aggregate voltage/thermal indicators are reported.

The next step is to add the iterative repair loop to these SimBench networks
only after the one-shot generalization behavior is understood.
""",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
