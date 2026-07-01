from __future__ import annotations

from pathlib import Path

import pandas as pd

import simbench_ai_load_generalization as base


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT / "datasets"
OUT = PACKAGE_ROOT / "results" / "00_analysis_outputs"


def main() -> None:
    original_hours = list(base.REPRESENTATIVE_HOURS)
    try:
        base.REPRESENTATIVE_HOURS = list(range(0, 24, 2))
        base.main()

        rename_pairs = [
            ("simbench_ai_generalization_networks.csv", "simbench_ai_generalization_extended_12h_networks.csv"),
            ("simbench_ai_generalization_dispatch.csv", "simbench_ai_generalization_extended_12h_dispatch.csv"),
            ("simbench_ai_generalization_results.csv", "simbench_ai_generalization_extended_12h_results.csv"),
            ("simbench_ai_generalization_summary.csv", "simbench_ai_generalization_extended_12h_summary.csv"),
        ]
        for src_name, dst_name in rename_pairs:
            src = OUT / src_name
            dst = OUT / dst_name
            if src.exists():
                dst.write_bytes(src.read_bytes())

        summary = pd.read_csv(OUT / "simbench_ai_generalization_extended_12h_summary.csv")
        log = f"""# SimBench Extended 12-Hour Generalization Log

## Purpose

This experiment extends the previous 8-representative-hour SimBench screening to
12 representative hours sampled every two hours from the moderate AI-load
scenario. It keeps the same candidate networks, AI-load scaling, storage sizing
rule, and deterministic AC power-flow certificate.

## Summary

```csv
{summary.to_csv(index=False)}```

## Interpretation

The extended run is a stronger generalization check than the initial 8-hour
sample while remaining stable on the local machine. It should be used to report
whether the proposed
storage/UPS dispatch and verification workflow remains effective across all
representative hours. A full 24-hour run can be launched later as a longer
batch experiment.
"""
        (OUT / "SIMBENCH_EXTENDED_12H_LOG.md").write_text(log, encoding="utf-8")
        print(summary.to_string(index=False))
    finally:
        base.REPRESENTATIVE_HOURS = original_hours


if __name__ == "__main__":
    main()
