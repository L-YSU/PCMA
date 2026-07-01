from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimize_ai_storage_dispatch import (
    OUT,
    FIG,
    build_stress_weight,
    build_voltage_safe_grid_upper,
    solve_storage_dispatch,
    summarize,
    verify_dispatch,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT / "datasets"


def score_verification(result_frame):
    min_v = float(result_frame["min_voltage_pu"].min())
    uv_hours = float(result_frame["undervoltage_bus_count_lt_0p95"].sum())
    loss = float(result_frame["line_losses_mw"].mean())
    peak = float(result_frame["ai_load_grid_seen_mw"].max())
    # Lexicographic-like scalar: prioritize physical violations first.
    return 1000.0 * uv_hours + 10000.0 * max(0.0, 0.95 - min_v) + 10.0 * loss + peak


def update_repair_signals(stress, grid_upper, dispatch, verification, raw_ai):
    verification = verification.sort_values("hour")
    grid = dispatch["optimized_grid_ai_mw"].to_numpy(dtype=float)
    min_v = verification["min_voltage_pu"].to_numpy(dtype=float)
    uv_count = verification["undervoltage_bus_count_lt_0p95"].to_numpy(dtype=float)
    gap = np.maximum(0.0, 0.95 - min_v)

    # Increase pressure weights for hours with actual physics violations.
    new_stress = np.asarray(stress, dtype=float).copy()
    new_stress *= 1.0 + 0.15 * uv_count + 45.0 * gap
    new_stress = np.clip(new_stress, 0.1, np.quantile(new_stress, 0.95) * 3.0)

    # Tighten the soft voltage ceiling where verification says the schedule is unsafe.
    reduction = 2.5 * gap + 0.004 * uv_count
    new_upper = np.asarray(grid_upper, dtype=float).copy()
    unsafe = gap > 0.0
    new_upper[unsafe] = np.minimum(new_upper[unsafe], grid[unsafe] - reduction[unsafe])
    new_upper = np.clip(new_upper, 0.0, float(np.max(raw_ai)))
    return new_stress, new_upper


def run_iterative_bus(ai_bus, max_iter=8):
    scenario = pd.read_csv(OUT / "ieee33_ai_datacenter_24h_scenario_moderate.csv")
    raw_ai = scenario["ai_load_mw"].to_numpy(dtype=float)
    price = scenario.get("price_pu", pd.Series(np.ones(len(scenario)))).to_numpy(dtype=float)
    stress = build_stress_weight(ai_bus, scenario)
    grid_upper = build_voltage_safe_grid_upper(ai_bus, scenario, raw_ai)

    iteration_summaries = []
    dispatches = []
    verifications = []
    best = None

    for iteration in range(max_iter):
        dispatch = solve_storage_dispatch(
            raw_ai,
            stress,
            price,
            grid_upper_mw=grid_upper,
        )
        dispatch["ai_bus"] = ai_bus
        dispatch["iteration"] = iteration
        verification = verify_dispatch(
            ai_bus,
            {"iterative_physics_certified": dispatch["optimized_grid_ai_mw"].to_numpy(dtype=float)},
            scenario,
        )
        verification["iteration"] = iteration
        score = score_verification(verification)

        summary = summarize(verification)
        summary["iteration"] = iteration
        summary["score"] = score
        summary["soft_excess_sum_mw"] = float(dispatch["voltage_limit_soft_excess_mw"].sum())
        summary["soft_excess_max_mw"] = float(dispatch["voltage_limit_soft_excess_mw"].max())
        iteration_summaries.append(summary)
        dispatches.append(dispatch)
        verifications.append(verification)

        if best is None or score < best["score"]:
            best = {
                "score": score,
                "iteration": iteration,
                "dispatch": dispatch.copy(),
                "verification": verification.copy(),
            }

        stress, grid_upper = update_repair_signals(
            stress,
            grid_upper,
            dispatch,
            verification,
            raw_ai,
        )

    all_dispatch = pd.concat(dispatches, ignore_index=True)
    all_verification = pd.concat(verifications, ignore_index=True)
    iter_summary = pd.concat(iteration_summaries, ignore_index=True)
    return all_dispatch, all_verification, iter_summary, best


def make_iterative_figures(best_by_bus, iter_summary):
    FIG.mkdir(exist_ok=True)
    for ai_bus, best in best_by_bus.items():
        dispatch = best["dispatch"]
        verification = best["verification"].sort_values("hour")
        plt.figure(figsize=(10, 4))
        plt.plot(dispatch["hour"], dispatch["raw_ai_mw"], marker="o", label="raw AI")
        plt.plot(
            dispatch["hour"],
            dispatch["optimized_grid_ai_mw"],
            marker="s",
            label=f"iterative certified, iter {best['iteration']}",
        )
        plt.bar(
            dispatch["hour"],
            dispatch["discharge_mw"] - dispatch["charge_mw"],
            alpha=0.25,
            label="storage net discharge",
        )
        plt.xlabel("Hour")
        plt.ylabel("MW")
        plt.title(f"Iterative Physics-Certified Dispatch, Bus {ai_bus}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / f"iterative_dispatch_bus_{ai_bus}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(
            verification["hour"],
            verification["min_voltage_pu"],
            marker="o",
            label="verified min voltage",
        )
        plt.axhline(0.95, color="red", linestyle="--", linewidth=1, label="0.95 p.u.")
        plt.xlabel("Hour")
        plt.ylabel("Minimum voltage (p.u.)")
        plt.title(f"Verified Voltage after Iterative Repair, Bus {ai_bus}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / f"iterative_verified_voltage_bus_{ai_bus}.png", dpi=200)
        plt.close()

    plt.figure(figsize=(9, 4))
    for ai_bus in sorted(iter_summary["ai_bus"].unique()):
        d = iter_summary[iter_summary["ai_bus"].eq(ai_bus)].sort_values("iteration")
        plt.plot(
            d["iteration"],
            d["undervoltage_bus_hours"],
            marker="o",
            label=f"bus {ai_bus}",
        )
    plt.xlabel("Repair iteration")
    plt.ylabel("Undervoltage bus-hours")
    plt.title("Physics-Certified Repair Progress")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "iterative_repair_progress.png", dpi=200)
    plt.close()


def write_iterative_log(iter_summary, final_summary, best_by_bus):
    best_lines = []
    for ai_bus, best in best_by_bus.items():
        best_lines.append(
            f"- bus {ai_bus}: best iteration {best['iteration']}, score {best['score']:.3f}"
        )
    (OUT / "STEP3_ITERATIVE_REPAIR_LOG.md").write_text(
        f"""# Step 3 Iterative Physics-Certified Repair Log

## Purpose

This experiment upgrades the one-shot storage/UPS dispatch into an
optimization-verification-repair loop.

Each iteration follows:

1. Solve a storage dispatch problem.
2. Verify the schedule with pandapower AC power flow.
3. Detect hours with undervoltage or large voltage deficit.
4. Increase stress weights and tighten voltage-safe grid-load ceilings.
5. Re-optimize.

## Best iterations

{chr(10).join(best_lines)}

## Iteration Summary

```csv
{iter_summary.to_csv(index=False)}```

## Final Comparison

```csv
{final_summary.to_csv(index=False)}```

## Research Meaning

This is closer to a high-standard power-system contribution than a one-shot dispatch model because
the acceptance condition is not decided by an LLM or by the optimizer objective.
It is decided by deterministic physical verification.

The current loop still uses a voltage-sensitivity proxy, so the next theoretical
step is to replace the proxy with topology-aware sensitivity factors or an
AC-feasibility cut generated from failed pandapower checks.
""",
        encoding="utf-8",
    )


def main():
    OUT.mkdir(exist_ok=True)
    all_dispatches = []
    all_verifications = []
    all_iter_summaries = []
    best_by_bus = {}
    scenario = pd.read_csv(OUT / "ieee33_ai_datacenter_24h_scenario_moderate.csv")
    raw_ai = scenario["ai_load_mw"].to_numpy(dtype=float)

    for ai_bus in [18, 33]:
        dispatches, verifications, iter_summary, best = run_iterative_bus(ai_bus)
        all_dispatches.append(dispatches)
        all_verifications.append(verifications)
        all_iter_summaries.append(iter_summary)
        best_by_bus[ai_bus] = best
        best["dispatch"].to_csv(
            OUT / f"iterative_best_dispatch_bus_{ai_bus}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    all_dispatch = pd.concat(all_dispatches, ignore_index=True)
    all_verification = pd.concat(all_verifications, ignore_index=True)
    iter_summary = pd.concat(all_iter_summaries, ignore_index=True)
    all_dispatch.to_csv(
        OUT / "iterative_all_dispatches.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_verification.to_csv(
        OUT / "iterative_all_verifications.csv",
        index=False,
        encoding="utf-8-sig",
    )
    iter_summary.to_csv(
        OUT / "iterative_repair_iteration_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_frames = []
    for ai_bus, best in best_by_bus.items():
        modes = {
            "no_ai": np.zeros_like(raw_ai),
            "raw_ai": raw_ai,
            "iterative_physics_certified": best["dispatch"][
                "optimized_grid_ai_mw"
            ].to_numpy(dtype=float),
        }
        final_frames.append(verify_dispatch(ai_bus, modes, scenario))
    final_results = pd.concat(final_frames, ignore_index=True)
    final_results.to_csv(
        OUT / "ieee33_iterative_physics_certified_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    final_summary = summarize(final_results)
    final_summary.to_csv(
        OUT / "ieee33_iterative_physics_certified_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    make_iterative_figures(best_by_bus, iter_summary)
    write_iterative_log(iter_summary, final_summary, best_by_bus)
    print(final_summary.to_string(index=False))
    print("\nIteration summary:")
    print(iter_summary.to_string(index=False))


if __name__ == "__main__":
    main()
