from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import simbench_ai_load_generalization as base
from optimize_ai_storage_dispatch import solve_storage_dispatch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT / "datasets"
OUT = PACKAGE_ROOT / "results" / "00_analysis_outputs"


def build_voltage_thermal_stress_and_upper(no_ai, raw_ai, raw_profile, background):
    stress, upper = base.build_stress_and_upper(no_ai, raw_ai, raw_profile, background)
    raw_ai = raw_ai.sort_values("hour")
    max_loading = raw_ai["max_line_loading_percent"].to_numpy(dtype=float)
    overload_count = raw_ai["overloaded_line_count"].to_numpy(dtype=float)
    thermal_excess = np.maximum(0.0, max_loading - 100.0) / 100.0

    # Thermal stress supplements the voltage stress. It is intentionally simple:
    # the schedule is still certified by AC power flow, not by this proxy.
    thermal_stress = 1.0 + 4.0 * thermal_excess + 0.08 * overload_count
    new_stress = np.asarray(stress, dtype=float) * thermal_stress

    new_upper = np.asarray(upper, dtype=float).copy()
    overloaded = max_loading > 100.0
    safe_fraction = np.ones_like(max_loading, dtype=float)
    safe_fraction[overloaded] = np.clip(98.0 / max_loading[overloaded], 0.0, 1.0)
    new_upper[overloaded] = np.minimum(new_upper[overloaded], np.asarray(raw_profile)[overloaded] * safe_fraction[overloaded])
    return new_stress, np.clip(new_upper, 0.0, float(np.max(raw_profile)))


def run_thermal_aware() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_norm, price, background = base.load_scenario()
    all_results = []
    dispatches = []
    network_records = []

    for code in base.CANDIDATE_CODES:
        net = base.prepare_net(code)
        ai_bus = base.choose_ai_bus(net)
        raw_ai = base.scale_ai_profile(net, raw_norm, fraction_of_base_load=0.015)

        no_ai = base.run_timeseries_verification(code, ai_bus, np.zeros_like(raw_ai), background, "no_ai")
        raw = base.run_timeseries_verification(code, ai_bus, raw_ai, background, "raw_ai")
        stress, upper = build_voltage_thermal_stress_and_upper(no_ai, raw, raw_ai, background)

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

        thermal_opt = base.run_timeseries_verification(
            code,
            ai_bus,
            dispatch["optimized_grid_ai_mw"].to_numpy(dtype=float),
            background,
            "thermal_aware_storage",
        )
        all_results.extend([no_ai, raw, thermal_opt])
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
    networks = pd.DataFrame(network_records)
    summary = base.summarize(results)
    return networks, dispatch, results, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    networks, dispatch, results, summary = run_thermal_aware()
    networks.to_csv(OUT / "simbench_thermal_aware_8h_networks.csv", index=False, encoding="utf-8-sig")
    dispatch.to_csv(OUT / "simbench_thermal_aware_8h_dispatch.csv", index=False, encoding="utf-8-sig")
    results.to_csv(OUT / "simbench_thermal_aware_8h_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "simbench_thermal_aware_8h_summary.csv", index=False, encoding="utf-8-sig")

    log = f"""# SimBench Thermal-Aware Repair Prototype

## Purpose

This prototype adds line-loading-aware repair signals to the SimBench storage
dispatch proxy. It is still a proxy; final safety is certified by pandapower AC
power flow.

## Summary

```csv
{summary.to_csv(index=False)}```

## Interpretation

This run tests the next algorithmic direction suggested by the 12-hour SimBench
extension: add thermal overload awareness rather than optimizing only voltage
stress. If thermal overload remains, the next step is to move from stress-weight
repair to explicit line-loading cuts or OPF-style constraints.
"""
    (OUT / "SIMBENCH_THERMAL_AWARE_8H_LOG.md").write_text(log, encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
