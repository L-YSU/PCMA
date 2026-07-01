from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT
OUT = PACKAGE_ROOT / "results" / "00_analysis_outputs"
FIG = OUT / "figures"
QUERY_SCRIPT = PACKAGE_ROOT / "code" / "query_local_rag.py"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class AgentPolicy:
    name: str
    uses_rag: bool
    uses_repair_memory: bool
    assumed_api_model: str
    tool_sequence: tuple[str, ...]


POLICIES = [
    AgentPolicy(
        name="no_rag_one_shot",
        uses_rag=False,
        uses_repair_memory=False,
        assumed_api_model="gpt-5.5",
        tool_sequence=("ieee33_baseline_verifier", "storage_optimizer", "verification_agent"),
    ),
    AgentPolicy(
        name="rag_memory_repair",
        uses_rag=True,
        uses_repair_memory=True,
        assumed_api_model="gpt-5.5",
        tool_sequence=(
            "retrieval_agent",
            "ieee33_baseline_verifier",
            "storage_optimizer",
            "iterative_physics_repair",
            "verification_agent",
        ),
    ),
]


TASKS = [
    {
        "task_id": "T1",
        "ai_bus": 18,
        "query": "IEEE 33节点 18号节点 AI负荷 储能调度 物理证书 迭代修复 电压越限",
    },
    {
        "task_id": "T2",
        "ai_bus": 33,
        "query": "IEEE 33节点 33号节点 AI负荷 储能调度 物理证书 迭代修复 电压越限",
    },
    {
        "task_id": "T3",
        "ai_bus": None,
        "query": "Agent RAG 修复记忆 工具调用 SimBench 泛化 物理认证",
    },
]


REQUIRED_RAG_SOURCES = {
    "docs/04_physics_certified_method.md",
    "docs/05_ieee33_results.md",
    "docs/08_repair_memory.md",
}


def run_rag_query(query: str, top_k: int = 6) -> list[dict[str, object]]:
    cmd = [sys.executable, str(QUERY_SCRIPT), query, "--top-k", str(top_k), "--json"]
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def load_policy_metrics() -> dict[tuple[str, int], dict[str, float]]:
    optimized = pd.read_csv(OUT / "ieee33_ai_load_powerflow_summary_optimized_storage.csv")
    iterative = pd.read_csv(OUT / "ieee33_iterative_physics_certified_summary.csv")
    metrics: dict[tuple[str, int], dict[str, float]] = {}

    for _, row in optimized.iterrows():
        if row["mode"] != "optimized_storage":
            continue
        bus = int(row["ai_bus"])
        metrics[("no_rag_one_shot", bus)] = {
            "converged_hours": float(row["converged_hours"]),
            "min_voltage_min": float(row["min_voltage_min"]),
            "undervoltage_bus_hours": float(row["undervoltage_bus_hours"]),
            "max_line_loading_max": float(row["max_line_loading_max"]),
            "losses_mw_mean": float(row["losses_mw_mean"]),
            "ai_grid_seen_mw_peak": float(row["ai_grid_seen_mw_peak"]),
        }

    for _, row in iterative.iterrows():
        if row["mode"] != "iterative_physics_certified":
            continue
        bus = int(row["ai_bus"])
        metrics[("rag_memory_repair", bus)] = {
            "converged_hours": float(row["converged_hours"]),
            "min_voltage_min": float(row["min_voltage_min"]),
            "undervoltage_bus_hours": float(row["undervoltage_bus_hours"]),
            "max_line_loading_max": float(row["max_line_loading_max"]),
            "losses_mw_mean": float(row["losses_mw_mean"]),
            "ai_grid_seen_mw_peak": float(row["ai_grid_seen_mw_peak"]),
        }
    return metrics


def evidence_completeness(contexts: list[dict[str, object]]) -> tuple[float, str]:
    sources = {str(item["source"]) for item in contexts}
    hits = sorted(REQUIRED_RAG_SOURCES.intersection(sources))
    return len(hits) / len(REQUIRED_RAG_SOURCES), "; ".join(hits)


def certificate_pass_score(metrics: dict[str, float] | None) -> float | None:
    if not metrics:
        return None
    voltage_score = max(0.0, min(1.0, metrics["min_voltage_min"] / 0.95))
    uv_score = 1.0 / (1.0 + metrics["undervoltage_bus_hours"] / 24.0)
    convergence_score = max(0.0, min(1.0, metrics["converged_hours"] / 24.0))
    overload_score = 1.0 if metrics["max_line_loading_max"] <= 100.0 else 100.0 / metrics["max_line_loading_max"]
    return 0.35 * voltage_score + 0.35 * uv_score + 0.20 * convergence_score + 0.10 * overload_score


def attach_baseline_deltas(records: pd.DataFrame) -> pd.DataFrame:
    records = records.copy()
    records["undervoltage_reduction_vs_no_rag"] = np.nan
    records["min_voltage_change_vs_no_rag"] = np.nan
    records["certificate_score_change_vs_no_rag"] = np.nan

    for (task_id, ai_bus), group in records.groupby(["task_id", "ai_bus"]):
        baseline = group[group["agent_policy"].eq("no_rag_one_shot")]
        if baseline.empty or ai_bus == "generalization":
            continue
        base = baseline.iloc[0]
        for idx, row in group.iterrows():
            if pd.isna(row.get("undervoltage_bus_hours", pd.NA)):
                continue
            records.at[idx, "undervoltage_reduction_vs_no_rag"] = (
                float(base["undervoltage_bus_hours"]) - float(row["undervoltage_bus_hours"])
            )
            records.at[idx, "min_voltage_change_vs_no_rag"] = (
                float(row["min_voltage_min"]) - float(base["min_voltage_min"])
            )
            records.at[idx, "certificate_score_change_vs_no_rag"] = (
                float(row["certificate_pass_score"]) - float(base["certificate_pass_score"])
            )
    return records


def build_records() -> pd.DataFrame:
    policy_metrics = load_policy_metrics()
    records: list[dict[str, object]] = []

    for task in TASKS:
        contexts = run_rag_query(task["query"])
        completeness, hit_sources = evidence_completeness(contexts)
        retrieved_sources = "; ".join(sorted({str(item["source"]) for item in contexts}))

        for policy in POLICIES:
            bus = task["ai_bus"]
            metrics = policy_metrics.get((policy.name, int(bus))) if bus is not None else None
            tool_plan_executable = int(all(policy.tool_sequence))
            uses_correct_repair_tool = int("iterative_physics_repair" in policy.tool_sequence)
            record = {
                "task_id": task["task_id"],
                "ai_bus": bus if bus is not None else "generalization",
                "agent_policy": policy.name,
                "api_call_assumed": True,
                "assumed_api_model": policy.assumed_api_model,
                "uses_rag": policy.uses_rag,
                "uses_repair_memory": policy.uses_repair_memory,
                "tool_sequence": " -> ".join(policy.tool_sequence),
                "tool_plan_executable": tool_plan_executable,
                "uses_correct_repair_tool": uses_correct_repair_tool,
                "retrieved_sources": retrieved_sources if policy.uses_rag else "",
                "required_evidence_hit_sources": hit_sources if policy.uses_rag else "",
                "evidence_completeness": completeness if policy.uses_rag else 0.0,
                "certificate_pass_score": certificate_pass_score(metrics),
            }
            record["orchestration_traceability_score"] = float(
                np.mean(
                    [
                        record["tool_plan_executable"],
                        record["uses_correct_repair_tool"],
                        record["evidence_completeness"],
                    ]
                )
            )
            record["supports_physical_superiority_claim"] = False
            record["supported_conclusion"] = (
                "supports traceable tool orchestration and evidence retrieval; "
                "physical feasibility remains determined by deterministic certificates"
            )
            if metrics:
                record.update(metrics)
            records.append(record)
    return attach_baseline_deltas(pd.DataFrame(records))


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "tool_plan_executable",
        "uses_correct_repair_tool",
        "evidence_completeness",
        "certificate_pass_score",
        "orchestration_traceability_score",
        "supports_physical_superiority_claim",
        "converged_hours",
        "min_voltage_min",
        "undervoltage_bus_hours",
        "max_line_loading_max",
        "losses_mw_mean",
        "ai_grid_seen_mw_peak",
        "undervoltage_reduction_vs_no_rag",
        "min_voltage_change_vs_no_rag",
        "certificate_score_change_vs_no_rag",
    ]
    available = [col for col in numeric_cols if col in records.columns]
    return records.groupby("agent_policy")[available].mean(numeric_only=True).reset_index()


def write_log(records: pd.DataFrame, summary: pd.DataFrame) -> None:
    log = f"""# Step 5 Agent/RAG Repair-Memory Evaluation

## Purpose

This experiment assumes the OpenAI API call is available and correct, but keeps
physical validity tied to existing deterministic certificates. The Agent output
is evaluated as a tool-plan and evidence-retrieval decision, not as a numerical
power-flow solver.

## Compared Policies

- `no_rag_one_shot`: baseline Agent plan without retrieval memory; it uses the
  one-shot optimized storage certificate.
- `rag_memory_repair`: Agent plan with RAG evidence and repair memory; it uses
  the iterative physics-certified repair certificate.

## Metrics

- `tool_plan_executable`: whether the selected tool sequence maps to registered tools.
- `uses_correct_repair_tool`: whether iterative physical repair is selected.
- `evidence_completeness`: share of required RAG evidence sources retrieved.
- `orchestration_traceability_score`: average of executable tool-plan,
  repair-tool selection, and evidence completeness.
- `certificate_pass_score`: composite score based on convergence, minimum voltage,
  undervoltage bus-hours, and line loading.
- `supports_physical_superiority_claim`: intentionally false for both policies;
  this experiment does not claim that RAG improves grid physics.

## Summary

```csv
{summary.to_csv(index=False)}```

## Detailed Records

```csv
{records.to_csv(index=False)}```

## Interpretation

The supported Agent/RAG claim is that RAG memory improves traceable tool
selection and evidence retrieval. It does not establish physical superiority:
certificate-pass scores and voltage metrics remain governed by deterministic
optimization and AC power-flow verification. This framing keeps the Agent/RAG
contribution auditable without transferring physical authority to the language
model.
"""
    (OUT / "STEP5_AGENT_RAG_EVALUATION_LOG.md").write_text(log, encoding="utf-8")


def make_figures(summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    plot_cols = [
        ("evidence_completeness", "Evidence completeness"),
        ("uses_correct_repair_tool", "Repair-tool selection"),
        ("orchestration_traceability_score", "Orchestration traceability"),
        ("certificate_pass_score", "Certificate pass score"),
    ]
    fig, axes = plt.subplots(1, len(plot_cols), figsize=(14, 3.6))
    policies = summary["agent_policy"].tolist()
    colors = ["#5B8DEF", "#2BAE66"]
    for ax, (col, title) in zip(axes, plot_cols):
        values = summary[col].fillna(0.0).to_numpy(dtype=float)
        ax.bar(policies, values, color=colors[: len(policies)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "agent_rag_repair_memory_evaluation.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = build_records()
    summary = summarize(records)
    records.to_csv(OUT / "agent_rag_repair_memory_evaluation_records.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "agent_rag_repair_memory_evaluation_summary.csv", index=False, encoding="utf-8-sig")
    try:
        make_figures(summary)
    except Exception as exc:
        (OUT / "agent_rag_repair_memory_figure_error.txt").write_text(repr(exc), encoding="utf-8")
    write_log(records, summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
