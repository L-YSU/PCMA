from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp

from optimize_ai_storage_dispatch import solve_storage_dispatch
from simbench_voltage_thermal_benchmark import (
    load_scenario,
    prepare_net,
    choose_ai_bus,
    scale_ai_profile,
    verify_profile,
    summarize,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUT = PACKAGE_ROOT / "results" / "00_analysis_outputs"

NETWORKS = [
    "1-HVMV-mixed-1.105-0-no_sw",
    "1-HVMV-mixed-2.102-0-no_sw",
]


def run_power_flow(net) -> None:
    pp.runpp(
        net,
        algorithm="nr",
        numba=False,
        max_iteration=100,
        calculate_voltage_angles=False,
        check_connectivity=True,
    )


def is_certificate_feasible(base_net, ai_bus: int, ai_mw: float, background: float) -> tuple[bool, float, float]:
    net = copy.deepcopy(base_net)
    if len(net.load):
        net.load.loc[:, "p_mw"] = net.load.p_mw.values * float(background)
        net.load.loc[:, "q_mvar"] = net.load.q_mvar.values * float(background)
    if ai_mw > 1e-9:
        pp.create_load(net, ai_bus, p_mw=float(ai_mw), q_mvar=0.25 * float(ai_mw), name="ai_datacenter_ac_ceiling")
    try:
        run_power_flow(net)
        min_v = float(net.res_bus.vm_pu.min())
        max_loading = float(net.res_line.loading_percent.max()) if len(net.line) else 0.0
        return bool(net.converged and min_v >= 0.95 and max_loading <= 100.0), min_v, max_loading
    except Exception:
        return False, np.nan, np.nan


def ac_feasibility_ceiling(base_net, ai_bus: int, raw_profile: np.ndarray, background: np.ndarray, n_iter: int = 18) -> pd.DataFrame:
    rows = []
    for step, raw_mw in enumerate(raw_profile):
        lo = 0.0
        hi = float(raw_mw)
        feasible_hi, hi_v, hi_l = is_certificate_feasible(base_net, ai_bus, hi, float(background[step]))
        if feasible_hi:
            limit = hi
            min_v = hi_v
            max_loading = hi_l
        else:
            min_v = np.nan
            max_loading = np.nan
            for _ in range(n_iter):
                mid = 0.5 * (lo + hi)
                ok, mid_v, mid_l = is_certificate_feasible(base_net, ai_bus, mid, float(background[step]))
                if ok:
                    lo = mid
                    min_v = mid_v
                    max_loading = mid_l
                else:
                    hi = mid
            limit = lo
        rows.append(
            {
                "step": step,
                "raw_ai_mw": float(raw_mw),
                "ac_feasible_grid_upper_mw": float(limit),
                "ceiling_fraction_of_raw": float(limit / max(float(raw_mw), 1e-9)),
                "ceiling_min_voltage_pu": float(min_v) if np.isfinite(min_v) else np.nan,
                "ceiling_max_line_loading_percent": float(max_loading) if np.isfinite(max_loading) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_strong_baseline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hours = list(range(24))
    raw_norm, price, background, _scenario = load_scenario(hours)
    all_results = []
    all_dispatches = []
    all_ceilings = []
    network_rows = []

    for code in NETWORKS:
        base_net = prepare_net(code)
        ai_bus = choose_ai_bus(base_net)
        raw_ai = scale_ai_profile(base_net, raw_norm)

        no_ai = verify_profile(code, base_net, ai_bus, np.zeros_like(raw_ai), background, "no_ai")
        raw = verify_profile(code, base_net, ai_bus, raw_ai, background, "raw_ai")
        ceiling = ac_feasibility_ceiling(base_net, ai_bus, raw_ai, background)

        stress = 1.0 + 20.0 * np.maximum(0.0, raw_ai - ceiling["ac_feasible_grid_upper_mw"].to_numpy(dtype=float)) / max(float(raw_ai.max()), 1e-9)
        dispatch = solve_storage_dispatch(
            raw_ai,
            stress,
            price,
            grid_upper_mw=ceiling["ac_feasible_grid_upper_mw"].to_numpy(dtype=float),
            battery_power_mw=max(0.05, 0.50 * float(raw_ai.max())),
            battery_capacity_mwh=max(0.2, 2.5 * float(raw_ai.max())),
            initial_soc_mwh=max(0.1, 1.25 * float(raw_ai.max())),
        )
        dispatch["code"] = code
        dispatch["ai_bus"] = ai_bus
        dispatch["strategy"] = "ac_feasibility_ceiling_storage"

        storage_limited = verify_profile(
            code,
            base_net,
            ai_bus,
            dispatch["optimized_grid_ai_mw"].to_numpy(dtype=float),
            background,
            "ac_feasibility_ceiling_storage",
        )
        oracle = verify_profile(
            code,
            base_net,
            ai_bus,
            ceiling["ac_feasible_grid_upper_mw"].to_numpy(dtype=float),
            background,
            "ac_feasibility_ceiling_oracle",
        )
        ceiling["code"] = code
        ceiling["ai_bus"] = ai_bus
        all_ceilings.append(ceiling)
        all_dispatches.append(dispatch)
        all_results.extend([no_ai, raw, storage_limited, oracle])
        network_rows.append(
            {
                "code": code,
                "ai_bus": ai_bus,
                "bus_count": len(base_net.bus),
                "line_count": len(base_net.line),
                "base_load_mw": float(base_net.load.p_mw.sum()) if len(base_net.load) else 0.0,
                "ai_peak_mw": float(raw_ai.max()),
            }
        )

    results = pd.concat(all_results, ignore_index=True)
    summary = summarize(results)
    return pd.DataFrame(network_rows), pd.concat(all_ceilings, ignore_index=True), pd.concat(all_dispatches, ignore_index=True), results, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    networks, ceilings, dispatch, results, summary = run_strong_baseline()
    networks.to_csv(OUT / "strong_baseline_ac_ceiling_networks.csv", index=False, encoding="utf-8-sig")
    ceilings.to_csv(OUT / "strong_baseline_ac_ceiling_limits.csv", index=False, encoding="utf-8-sig")
    dispatch.to_csv(OUT / "strong_baseline_ac_ceiling_dispatch.csv", index=False, encoding="utf-8-sig")
    results.to_csv(OUT / "strong_baseline_ac_ceiling_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "strong_baseline_ac_ceiling_summary.csv", index=False, encoding="utf-8-sig")
    log = f"""# Strong Baseline: AC Feasibility-Ceiling Storage

This baseline computes an AC power-flow certificate-guided upper bound on
grid-seen AI load at every hour by bisection.

Two reference modes are reported:

- `ac_feasibility_ceiling_storage`: a service-preserving storage dispatch under
  the same power, energy, and terminal-SOC assumptions as the main storage
  experiments.
- `ac_feasibility_ceiling_oracle`: a non-deployable feasibility reference that
  directly injects the hourly AC-feasible ceiling. It is a strong lower-bound
  reference for what voltage and thermal violations could look like if the AI
  data-center net load could be shaped to the certified ceiling.

Both profiles are verified again by independent AC power flow.

```csv
{summary.to_csv(index=False)}```
"""
    (OUT / "STRONG_BASELINE_AC_CEILING_LOG.md").write_text(log, encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
