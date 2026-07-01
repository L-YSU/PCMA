from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT / "datasets"
OUT = PACKAGE_ROOT / "results" / "00_analysis_outputs"
FIG = OUT / "figures"


def normalize(values):
    values = np.asarray(values, dtype=float)
    lo = values.min()
    hi = values.max()
    if hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def metrics(name, series):
    values = np.asarray(series, dtype=float)
    diff = np.diff(values)
    p05 = np.quantile(values, 0.05)
    return {
        "scenario": name,
        "points": len(values),
        "min_mw": float(values.min()),
        "mean_mw": float(values.mean()),
        "p95_mw": float(np.quantile(values, 0.95)),
        "max_mw": float(values.max()),
        "peak_to_mean": float(values.max() / max(values.mean(), 1e-9)),
        "peak_to_idle": float(values.max() / max(p05, 1e-9)),
        "max_ramp_mw_per_min": float(diff.max()) if len(diff) else 0.0,
        "max_drop_mw_per_min": float(diff.min()) if len(diff) else 0.0,
    }


def main():
    FIG.mkdir(exist_ok=True)
    rep = pd.read_csv(OUT / "representative_ai_load_24h_1min.csv")
    base = normalize(rep["raw_power_w"].to_numpy(dtype=float))
    minutes = np.arange(len(base))
    rng = np.random.default_rng(42)

    # Training: sustained high load with low-frequency variations.
    training = 0.55 + 0.35 * pd.Series(base).rolling(45, min_periods=1).mean().to_numpy()
    training += 0.04 * np.sin(2 * np.pi * minutes / 360.0)
    training = np.clip(training, 0.2, 1.0)
    training_mw = 0.4 + 0.9 * training

    # Fine-tuning: staged bursts with evaluation valleys.
    finetune = 0.25 + 0.65 * base
    eval_mask = ((minutes // 120) % 3) == 2
    finetune[eval_mask] *= 0.55
    finetune += 0.03 * rng.normal(size=len(finetune))
    finetune = np.clip(finetune, 0.05, 1.0)
    finetune_mw = 0.15 + 0.85 * finetune

    # Inference burst: sparse sharp peaks driven by user demand.
    inference = 0.04 + 0.10 * base
    burst_centers = [75, 142, 355, 515, 730, 910, 1045, 1230, 1350]
    for center in burst_centers:
        inference += 0.95 * np.exp(-0.5 * ((minutes - center) / 8.0) ** 2)
    inference = np.clip(inference, 0.02, 1.0)
    inference_mw = 0.05 + 0.75 * inference

    # RAG batch: retrieval plateau plus generation peaks.
    rag = 0.10 + 0.22 * base
    for start in [180, 420, 780, 1080]:
        rag[start : start + 80] += 0.28
        for center in [start + 15, start + 42, start + 68]:
            rag += 0.32 * np.exp(-0.5 * ((minutes - center) / 5.0) ** 2)
    rag = np.clip(rag, 0.05, 1.0)
    rag_mw = 0.08 + 0.82 * rag

    scenarios = {
        "training_sustained": training_mw,
        "finetune_stagewise": finetune_mw,
        "inference_burst": inference_mw,
        "rag_batch": rag_mw,
    }

    rows = []
    metric_rows = []
    hourly_rows = []
    for name, values in scenarios.items():
        metric_rows.append(metrics(name, values))
        for minute, value in enumerate(values):
            rows.append({"scenario": name, "minute": minute, "ai_load_mw": value})
        hourly = pd.Series(values).groupby(minutes // 60).mean()
        for hour, value in hourly.items():
            hourly_rows.append({"scenario": name, "hour": int(hour), "ai_load_mw": float(value)})

    pd.DataFrame(rows).to_csv(
        OUT / "ai_workload_scenario_library_1min.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(hourly_rows).to_csv(
        OUT / "ai_workload_scenario_library_hourly.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(
        OUT / "ai_workload_scenario_library_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(11, 5))
    for name, values in scenarios.items():
        plt.plot(minutes / 60.0, values, linewidth=1.2, label=name)
    plt.xlabel("Hour")
    plt.ylabel("MW")
    plt.title("AI Workload Scenario Library Derived from MIT Supercloud Trace")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "ai_workload_scenario_library.png", dpi=200)
    plt.close()

    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
