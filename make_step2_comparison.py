from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT / "datasets"
OUT = PACKAGE_ROOT / "results" / "00_analysis_outputs"
FIG = OUT / "figures"


def main():
    stage1 = pd.read_csv(OUT / "ieee33_ai_load_powerflow_summary_moderate.csv")
    stage2 = pd.read_csv(OUT / "ieee33_ai_load_powerflow_summary_optimized_storage.csv")
    iterative_path = OUT / "ieee33_iterative_physics_certified_summary.csv"
    keep_stage1 = stage1[stage1["mode"].isin(["no_ai", "raw_ai", "smoothed_ai"])]
    keep_stage2 = stage2[stage2["mode"].eq("optimized_storage")]
    frames = [keep_stage1, keep_stage2]
    if iterative_path.exists():
        iterative = pd.read_csv(iterative_path)
        frames.append(iterative[iterative["mode"].eq("iterative_physics_certified")])
    combined = pd.concat(frames, ignore_index=True)
    order = {
        "no_ai": 0,
        "raw_ai": 1,
        "smoothed_ai": 2,
        "optimized_storage": 3,
        "iterative_physics_certified": 4,
    }
    combined["mode_order"] = combined["mode"].map(order)
    combined = combined.sort_values(["ai_bus", "mode_order"]).drop(columns=["mode_order"])
    combined.to_csv(
        OUT / "ieee33_stage1_stage2_combined_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    FIG.mkdir(exist_ok=True)
    for metric, ylabel, filename in [
        ("min_voltage_min", "Worst minimum voltage (p.u.)", "combined_worst_voltage.png"),
        (
            "undervoltage_bus_hours",
            "Undervoltage bus-hours",
            "combined_undervoltage_bus_hours.png",
        ),
        ("losses_mw_mean", "Mean line losses (MW)", "combined_mean_losses.png"),
    ]:
        plt.figure(figsize=(9, 4))
        for i, bus in enumerate([18, 33]):
            data = combined[combined["ai_bus"].eq(bus)]
            x = range(len(data))
            offset = -0.18 if bus == 18 else 0.18
            plt.bar([v + offset for v in x], data[metric], width=0.35, label=f"bus {bus}")
        labels = combined[combined["ai_bus"].eq(18)]["mode"].tolist()
        plt.xticks(range(len(labels)), labels, rotation=15)
        plt.ylabel(ylabel)
        plt.title(ylabel + " by control mode")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / filename, dpi=200)
        plt.close()

    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
