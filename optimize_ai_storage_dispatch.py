from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandapower as pp
from scipy.optimize import linprog


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT / "datasets"
IEEE = ROOT / "IEEE_33_bus_Baran_Wu"
OUT = PACKAGE_ROOT / "results" / "00_analysis_outputs"
FIG = OUT / "figures"


def build_ieee33(slack_vm_pu=1.03):
    buses = pd.read_csv(IEEE / "bus_loads.csv")
    branches = pd.read_csv(IEEE / "branches.csv")
    net = pp.create_empty_network(sn_mva=100.0)
    bus_map = {
        int(bus): pp.create_bus(net, vn_kv=12.66, name=f"bus_{int(bus)}")
        for bus in buses["bus"]
    }
    pp.create_ext_grid(
        net,
        bus_map[1],
        vm_pu=slack_vm_pu,
        name="substation_voltage_regulated",
    )
    for _, row in branches.iterrows():
        pp.create_line_from_parameters(
            net,
            bus_map[int(row["from_bus"])],
            bus_map[int(row["to_bus"])],
            length_km=1.0,
            r_ohm_per_km=float(row["r_ohm"]),
            x_ohm_per_km=float(row["x_ohm"]),
            c_nf_per_km=0.0,
            max_i_ka=0.4,
            name=f"{int(row['from_bus'])}-{int(row['to_bus'])}",
        )
    for _, row in buses.iterrows():
        if int(row["bus"]) == 1:
            continue
        pp.create_load(
            net,
            bus_map[int(row["bus"])],
            p_mw=float(row["p_kw"]) / 1000.0,
            q_mvar=float(row["q_kvar"]) / 1000.0,
            name=f"load_{int(row['bus'])}",
        )
    return net, bus_map


def solve_storage_dispatch(
    raw_ai_mw,
    stress_weight,
    price_weight,
    grid_upper_mw=None,
    battery_power_mw=0.75,
    battery_capacity_mwh=3.0,
    initial_soc_mwh=1.5,
    eta_charge=0.95,
    eta_discharge=0.95,
):
    """Linear storage dispatch.

    Variables per hour:
      grid[t], charge[t], discharge[t], soc[t], ramp_pos[t], ramp_neg[t]
    plus one peak variable.

    The AI service is fully preserved:
      grid[t] = raw_ai[t] + charge[t] - discharge[t]
    """

    raw = np.asarray(raw_ai_mw, dtype=float)
    stress = np.asarray(stress_weight, dtype=float)
    price = np.asarray(price_weight, dtype=float)
    t_count = len(raw)
    dt_h = 1.0

    n_block = t_count
    idx_grid = 0
    idx_charge = idx_grid + n_block
    idx_discharge = idx_charge + n_block
    idx_soc = idx_discharge + n_block
    idx_ramp_pos = idx_soc + n_block
    idx_ramp_neg = idx_ramp_pos + n_block
    idx_voltage_excess = idx_ramp_neg + n_block
    idx_peak = idx_voltage_excess + n_block
    n_var = idx_peak + 1

    c = np.zeros(n_var)
    stress_norm = (stress - stress.min()) / (stress.max() - stress.min() + 1e-9)
    price_norm = (price - price.min()) / (price.max() - price.min() + 1e-9)
    c[idx_grid : idx_grid + t_count] = 10.0 * stress_norm + 0.8 * price_norm
    c[idx_charge : idx_charge + t_count] = 0.05
    c[idx_discharge : idx_discharge + t_count] = 0.05
    c[idx_ramp_pos : idx_ramp_pos + t_count] = 1.2
    c[idx_ramp_neg : idx_ramp_neg + t_count] = 1.2
    c[idx_voltage_excess : idx_voltage_excess + t_count] = 200.0
    c[idx_peak] = 5.0

    if grid_upper_mw is None:
        grid_upper = np.full(t_count, float(raw.max()))
    else:
        grid_upper = np.asarray(grid_upper_mw, dtype=float).copy()
        grid_upper = np.minimum(grid_upper, float(raw.max()))
        grid_upper = np.maximum(grid_upper, 0.0)
    high_stress_hours = stress_norm >= np.quantile(stress_norm, 0.60)
    grid_upper[high_stress_hours] = np.minimum(grid_upper[high_stress_hours], raw[high_stress_hours])

    bounds = []
    bounds.extend((0.0, float(raw.max())) for _ in range(t_count))  # grid
    bounds.extend((0.0, battery_power_mw) for _ in range(t_count))  # charge
    bounds.extend((0.0, battery_power_mw) for _ in range(t_count))  # discharge
    bounds.extend((0.05 * battery_capacity_mwh, battery_capacity_mwh) for _ in range(t_count))
    bounds.extend((0.0, None) for _ in range(t_count))  # ramp positive
    bounds.extend((0.0, None) for _ in range(t_count))  # ramp negative
    bounds.extend((0.0, None) for _ in range(t_count))  # voltage-limit soft excess
    bounds.append((0.0, float(raw.max())))  # peak

    a_eq = []
    b_eq = []

    # grid[t] - charge[t] + discharge[t] = raw[t]
    for t in range(t_count):
        row = np.zeros(n_var)
        row[idx_grid + t] = 1.0
        row[idx_charge + t] = -1.0
        row[idx_discharge + t] = 1.0
        a_eq.append(row)
        b_eq.append(raw[t])

    # SOC recursion.
    for t in range(t_count):
        row = np.zeros(n_var)
        row[idx_soc + t] = 1.0
        if t > 0:
            row[idx_soc + t - 1] = -1.0
            rhs = 0.0
        else:
            rhs = initial_soc_mwh
        row[idx_charge + t] = -eta_charge * dt_h
        row[idx_discharge + t] = dt_h / eta_discharge
        a_eq.append(row)
        b_eq.append(rhs)

    # Terminal SOC neutrality for fair comparison.
    row = np.zeros(n_var)
    row[idx_soc + t_count - 1] = 1.0
    a_eq.append(row)
    b_eq.append(initial_soc_mwh)

    a_ub = []
    b_ub = []

    # Peak constraint: grid[t] <= peak.
    for t in range(t_count):
        row = np.zeros(n_var)
        row[idx_grid + t] = 1.0
        row[idx_peak] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)

    # Physics-guided soft ceiling: grid[t] <= voltage-safe upper[t] + excess[t].
    for t in range(t_count):
        row = np.zeros(n_var)
        row[idx_grid + t] = 1.0
        row[idx_voltage_excess + t] = -1.0
        a_ub.append(row)
        b_ub.append(float(grid_upper[t]))

    # Absolute ramp representation.
    for t in range(1, t_count):
        row = np.zeros(n_var)
        row[idx_grid + t] = 1.0
        row[idx_grid + t - 1] = -1.0
        row[idx_ramp_pos + t] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)

        row = np.zeros(n_var)
        row[idx_grid + t - 1] = 1.0
        row[idx_grid + t] = -1.0
        row[idx_ramp_neg + t] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)

    result = linprog(
        c,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=np.asarray(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Storage dispatch optimization failed: {result.message}")

    x = result.x
    return pd.DataFrame(
        {
            "hour": np.arange(t_count),
            "raw_ai_mw": raw,
            "optimized_grid_ai_mw": x[idx_grid : idx_grid + t_count],
            "charge_mw": x[idx_charge : idx_charge + t_count],
            "discharge_mw": x[idx_discharge : idx_discharge + t_count],
            "soc_mwh": x[idx_soc : idx_soc + t_count],
            "voltage_safe_grid_upper_mw": grid_upper,
            "voltage_limit_soft_excess_mw": x[idx_voltage_excess : idx_voltage_excess + t_count],
            "stress_weight": stress,
            "price_weight": price,
        }
    )


def verify_dispatch(ai_bus_no, mode_to_ai_series, scenario):
    records = []
    for mode, ai_series in mode_to_ai_series.items():
        for h, row in scenario.iterrows():
            net, bus_map = build_ieee33()
            multiplier = float(row.get("background_load_multiplier", 1.0))
            net.load["p_mw"] *= multiplier
            net.load["q_mvar"] *= multiplier
            ai_mw = float(ai_series[h])
            if ai_mw > 0:
                pp.create_load(
                    net,
                    bus_map[ai_bus_no],
                    p_mw=ai_mw,
                    q_mvar=0.25 * ai_mw,
                    name=f"ai_datacenter_{mode}",
                )
            try:
                pp.runpp(
                    net,
                    algorithm="bfsw",
                    numba=False,
                    max_iteration=100,
                    tolerance_mva=1e-6,
                )
                converged = bool(net.converged)
                min_voltage = float(net.res_bus.vm_pu.min())
                max_line_loading = float(net.res_line.loading_percent.max())
                overloaded = int((net.res_line.loading_percent > 100.0).sum())
                undervoltage = int((net.res_bus.vm_pu < 0.95).sum())
                losses = float(net.res_line.pl_mw.sum())
                slack = float(net.res_ext_grid.p_mw.sum())
            except Exception:
                converged = False
                min_voltage = np.nan
                max_line_loading = np.nan
                overloaded = np.nan
                undervoltage = np.nan
                losses = np.nan
                slack = np.nan

            records.append(
                {
                    "ai_bus": ai_bus_no,
                    "mode": mode,
                    "hour": int(h),
                    "converged": converged,
                    "ai_load_grid_seen_mw": ai_mw,
                    "min_voltage_pu": min_voltage,
                    "max_line_loading_percent": max_line_loading,
                    "overloaded_line_count": overloaded,
                    "undervoltage_bus_count_lt_0p95": undervoltage,
                    "line_losses_mw": losses,
                    "slack_power_mw": slack,
                }
            )
    return pd.DataFrame(records)


def build_stress_weight(ai_bus_no, scenario):
    prev = pd.read_csv(OUT / "ieee33_ai_load_powerflow_results_moderate.csv")
    raw = prev[(prev["ai_bus"] == ai_bus_no) & (prev["mode"] == "raw_ai")].sort_values("hour")
    if len(raw) != len(scenario):
        base = scenario["background_load_multiplier"].to_numpy(dtype=float)
        return 1.0 + base

    voltage_gap = np.maximum(0.0, 0.95 - raw["min_voltage_pu"].to_numpy(dtype=float))
    undervoltage_count = raw["undervoltage_bus_count_lt_0p95"].to_numpy(dtype=float)
    base = scenario["background_load_multiplier"].to_numpy(dtype=float)
    return 1.0 + 25.0 * voltage_gap + 0.08 * undervoltage_count + 0.5 * base


def build_voltage_safe_grid_upper(ai_bus_no, scenario, raw_ai):
    prev = pd.read_csv(OUT / "ieee33_ai_load_powerflow_results_moderate.csv")
    no_ai = prev[(prev["ai_bus"] == ai_bus_no) & (prev["mode"] == "no_ai")].sort_values("hour")
    raw = prev[(prev["ai_bus"] == ai_bus_no) & (prev["mode"] == "raw_ai")].sort_values("hour")
    if len(no_ai) != len(scenario) or len(raw) != len(scenario):
        return np.full(len(scenario), float(np.max(raw_ai)))

    base_v = no_ai["min_voltage_pu"].to_numpy(dtype=float)
    raw_v = raw["min_voltage_pu"].to_numpy(dtype=float)
    p_raw = np.asarray(raw_ai, dtype=float)
    slope = (base_v - raw_v) / np.maximum(p_raw, 1e-6)

    # Estimate a conservative AI-load ceiling that keeps a small voltage reserve.
    voltage_limit = 0.95
    reserve = 0.002
    upper = np.full(len(scenario), float(np.max(p_raw)))
    mask = slope > 1e-6
    upper[mask] = (base_v[mask] - voltage_limit - reserve) / slope[mask]
    upper = np.clip(upper, 0.0, float(np.max(p_raw)))
    return upper


def summarize(results):
    return (
        results.groupby(["ai_bus", "mode"])
        .agg(
            converged_hours=("converged", "sum"),
            min_voltage_min=("min_voltage_pu", "min"),
            min_voltage_mean=("min_voltage_pu", "mean"),
            max_line_loading_max=("max_line_loading_percent", "max"),
            overloaded_line_hours=("overloaded_line_count", "sum"),
            undervoltage_bus_hours=("undervoltage_bus_count_lt_0p95", "sum"),
            losses_mw_mean=("line_losses_mw", "mean"),
            slack_mw_peak=("slack_power_mw", "max"),
            ai_grid_seen_mw_peak=("ai_load_grid_seen_mw", "max"),
        )
        .reset_index()
    )


def make_figures(all_dispatch, all_results, summary):
    FIG.mkdir(exist_ok=True)
    for ai_bus, dispatch in all_dispatch.items():
        plt.figure(figsize=(10, 4))
        plt.plot(dispatch["hour"], dispatch["raw_ai_mw"], marker="o", label="raw AI load")
        plt.plot(
            dispatch["hour"],
            dispatch["optimized_grid_ai_mw"],
            marker="s",
            label="grid-seen AI load after storage",
        )
        plt.bar(
            dispatch["hour"],
            dispatch["discharge_mw"] - dispatch["charge_mw"],
            alpha=0.25,
            label="storage net discharge",
        )
        plt.xlabel("Hour")
        plt.ylabel("MW")
        plt.title(f"Optimized AI Load Seen by Grid, AI at Bus {ai_bus}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / f"optimized_storage_dispatch_bus_{ai_bus}.png", dpi=200)
        plt.close()

        d = all_results[all_results["ai_bus"] == ai_bus]
        plt.figure(figsize=(10, 4))
        for mode in ["no_ai", "raw_ai", "optimized_storage"]:
            dd = d[d["mode"] == mode]
            plt.plot(dd["hour"], dd["min_voltage_pu"], marker="o", label=mode)
        plt.axhline(0.95, color="red", linestyle="--", linewidth=1, label="0.95 p.u.")
        plt.xlabel("Hour")
        plt.ylabel("Minimum voltage (p.u.)")
        plt.title(f"Physics Verification after Storage Optimization, Bus {ai_bus}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / f"optimized_min_voltage_bus_{ai_bus}.png", dpi=200)
        plt.close()

    plt.figure(figsize=(9, 4))
    modes = ["no_ai", "raw_ai", "optimized_storage"]
    x = np.arange(len(modes))
    for offset, bus in [(-0.18, 18), (0.18, 33)]:
        dd = summary[summary["ai_bus"] == bus].set_index("mode").loc[modes]
        plt.bar(x + offset, dd["undervoltage_bus_hours"], width=0.35, label=f"bus {bus}")
    plt.xticks(x, modes)
    plt.ylabel("Undervoltage bus-hours")
    plt.title("Undervoltage Reduction after Optimized Storage Dispatch")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "optimized_undervoltage_comparison.png", dpi=200)
    plt.close()


def write_log(dispatch_by_bus, summary):
    log_path = OUT / "STEP2_OPTIMIZATION_LOG.md"
    summary_csv = summary.to_csv(index=False)
    log_path.write_text(
        f"""# Step 2 Optimization Experiment Log

## Goal

Upgrade the Stage-1 rule-based smoothing experiment into a deterministic
storage/UPS optimization experiment, then verify every schedule with pandapower.

## Model

The AI workload demand is preserved hour by hour. The optimization only changes
what the upstream grid sees by charging or discharging a local UPS/storage
device:

`grid_ai_load[t] = raw_ai_load[t] + charge[t] - discharge[t]`

The storage state is constrained by power limit, energy capacity, round-trip
efficiency, and terminal SOC neutrality.

## Objective

The linear program penalizes:

- AI load drawn from the grid during electrically stressed hours;
- day-ahead price proxy from OPSD;
- peak grid-seen AI load;
- positive and negative ramping of grid-seen AI load;
- small charge/discharge degradation terms.

Stress weights are built from the Stage-1 raw-AI pandapower verification:
low voltage and many undervoltage buses receive higher weight.

After an infeasible hard-constraint attempt, the final version uses a
physics-guided soft voltage ceiling. The ceiling is estimated from the no-AI
and raw-AI pandapower results. Violating this ceiling is allowed but receives a
large penalty and is written as `voltage_limit_soft_excess_mw` in the dispatch
CSV files. This keeps the optimization feasible while exposing the remaining
physical stress.

## Storage Parameters

- Power limit: 0.75 MW
- Energy capacity: 3.0 MWh
- Initial and terminal SOC: 1.5 MWh
- Charge/discharge efficiency: 0.95

## Result Summary

```csv
{summary_csv}```

## Interpretation

This is still a simplified storage/UPS controller, but it establishes the
method's core closed loop:

real AI trace -> grid scenario -> optimization -> physical verification.

The next improvement is to replace the stress proxy with topology-aware voltage
sensitivity or an iterative optimization-verification loop.
""",
        encoding="utf-8",
    )


def main():
    OUT.mkdir(exist_ok=True)
    scenario = pd.read_csv(OUT / "ieee33_ai_datacenter_24h_scenario_moderate.csv")
    raw_ai = scenario["ai_load_mw"].to_numpy(dtype=float)
    price = scenario.get("price_pu", pd.Series(np.ones(len(scenario)))).to_numpy(dtype=float)

    dispatch_frames = {}
    verification_frames = []
    for ai_bus in [18, 33]:
        stress = build_stress_weight(ai_bus, scenario)
        grid_upper = build_voltage_safe_grid_upper(ai_bus, scenario, raw_ai)
        dispatch = solve_storage_dispatch(raw_ai, stress, price, grid_upper_mw=grid_upper)
        dispatch["ai_bus"] = ai_bus
        dispatch_frames[ai_bus] = dispatch
        dispatch.to_csv(
            OUT / f"optimized_storage_dispatch_bus_{ai_bus}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        mode_to_ai = {
            "no_ai": np.zeros_like(raw_ai),
            "raw_ai": raw_ai,
            "optimized_storage": dispatch["optimized_grid_ai_mw"].to_numpy(dtype=float),
        }
        verification_frames.append(verify_dispatch(ai_bus, mode_to_ai, scenario))

    results = pd.concat(verification_frames, ignore_index=True)
    results.to_csv(
        OUT / "ieee33_ai_load_powerflow_results_optimized_storage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = summarize(results)
    summary.to_csv(
        OUT / "ieee33_ai_load_powerflow_summary_optimized_storage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    make_figures(dispatch_frames, results, summary)
    write_log(dispatch_frames, summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
