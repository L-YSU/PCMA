from __future__ import annotations

import csv
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUT = PACKAGE_ROOT / "results" / "00_analysis_outputs"


CLAIMS = [
    {
        "claim_id": "C1",
        "claim": "Real AI/HPC traces can be converted into grid-side AI data-center load scenarios.",
        "claim_cn": "真实 AI/HPC 轨迹可以转化为面向电网分析的 AI 数据中心负荷场景。",
        "datasets": "MIT Supercloud sample",
        "scripts": "code/generate_ai_workload_scenario_library.py",
        "evidence_files": "results/00_analysis_outputs/ai_workload_scenario_library_metrics.csv; results/00_analysis_outputs/representative_ai_load_24h_1min.csv",
        "certificate": "data traceability and scenario metrics",
        "current_status": "supported",
        "next_needed": "Add more traces if available; report workload-class sensitivity.",
    },
    {
        "claim_id": "C2",
        "claim": "Raw AI load causes measurable distribution-feeder stress depending on interconnection location.",
        "claim_cn": "原始 AI 负荷会在不同接入位置造成可量化的配电网压力。",
        "datasets": "IEEE 33-bus; MIT-derived AI profile",
        "scripts": "code/run_ieee33_ai_load_powerflow_moderate.py",
        "evidence_files": "results/00_analysis_outputs/ieee33_ai_load_powerflow_summary_moderate.csv",
        "certificate": "pandapower AC power-flow convergence and voltage metrics",
        "current_status": "supported for buses 18 and 33",
        "next_needed": "Screen more candidate buses and include sensitivity to AI peak size.",
    },
    {
        "claim_id": "C3",
        "claim": "Storage/UPS dispatch can reduce grid-seen AI load stress relative to raw AI load.",
        "claim_cn": "储能/UPS 调度能够降低电网所见 AI 负荷压力。",
        "datasets": "IEEE 33-bus; MIT-derived AI profile",
        "scripts": "code/optimize_ai_storage_dispatch.py",
        "evidence_files": "results/00_analysis_outputs/ieee33_ai_load_powerflow_summary_optimized_storage.csv; results/00_analysis_outputs/optimized_storage_dispatch_bus_18.csv; results/00_analysis_outputs/optimized_storage_dispatch_bus_33.csv",
        "certificate": "linear dispatch plus post-hoc AC power-flow verification",
        "current_status": "partially supported",
        "next_needed": "Compare against stronger baselines and tune constraints for full certificate pass.",
    },
    {
        "claim_id": "C4",
        "claim": "An optimization-verification-repair loop is more defensible than one-shot LLM or one-shot optimization.",
        "claim_cn": "优化-验证-修复闭环比一次性 LLM 决策或一次性优化更适合高水平电力系统研究。",
        "datasets": "IEEE 33-bus; MIT-derived AI profile",
        "scripts": "code/iterative_physics_certified_dispatch.py",
        "evidence_files": "results/00_analysis_outputs/ieee33_iterative_physics_certified_summary.csv; results/00_analysis_outputs/iterative_repair_iteration_summary.csv",
        "certificate": "iterative pandapower counterexample repair",
        "current_status": "supported as method prototype",
        "next_needed": "Add ablation: no repair, heuristic repair, RAG memory repair, oracle repair.",
    },
    {
        "claim_id": "C5",
        "claim": "The workflow generalizes beyond IEEE 33-bus to SimBench topologies.",
        "claim_cn": "该流程可以从 IEEE 33-bus 推广到 SimBench 拓扑。",
        "datasets": "SimBench sample; MIT-derived AI profile",
        "scripts": "code/simbench_ai_load_generalization.py",
        "evidence_files": "results/00_analysis_outputs/simbench_ai_generalization_summary.csv; results/00_analysis_outputs/simbench_ai_generalization_extended_12h_summary.csv; results/00_analysis_outputs/SIMBENCH_EXTENDED_12H_LOG.md",
        "certificate": "multi-topology AC power-flow verification",
        "current_status": "supported on 12 representative hours for two SimBench networks",
        "next_needed": "Add thermal-overload repair cuts and then run full 24-hour batch.",
    },
    {
        "claim_id": "C6",
        "claim": "Agent/RAG can improve traceability and orchestration while keeping deterministic physical certificates authoritative.",
        "claim_cn": "Agent/RAG 能够提升可追溯性和工具编排能力，但物理证书仍由确定性工具给出。",
        "datasets": "RAG knowledge base from local experiment outputs",
        "scripts": "code/build_local_rag_index.py; code/query_local_rag.py; agent_rag/agent_rag_blueprint/openai_agent_api_client.py",
        "evidence_files": "agent_rag/rag_knowledge_base/index/manifest.json; results/00_analysis_outputs/agent_rag_api_response.json",
        "certificate": "retrieval provenance plus deterministic tool registry",
        "current_status": "prototype implemented",
        "next_needed": "Evaluate executable tool-plan rate and certificate pass rate with/without RAG memory.",
    },
    {
        "claim_id": "C7",
        "claim": "RAG repair memory improves tool-selection traceability, while physical performance must still be judged by deterministic certificates.",
        "claim_cn": "RAG 修复记忆提升工具选择与证据追踪能力，但不能单独支撑物理性能提升结论。",
        "datasets": "RAG knowledge base; IEEE 33-bus certificate outputs",
        "scripts": "code/evaluate_agent_rag_repair_memory.py",
        "evidence_files": "results/00_analysis_outputs/agent_rag_repair_memory_evaluation_summary.csv; results/00_analysis_outputs/STEP5_AGENT_RAG_EVALUATION_LOG.md",
        "certificate": "agent tool-plan metrics plus pandapower-derived certificate metrics",
        "current_status": "supported for orchestration and traceability only",
        "next_needed": "Keep Agent metrics separate from deterministic power-system certificate metrics.",
    },
    {
        "claim_id": "C8",
        "claim": "Thermal-aware repair addresses the key limitation found in SimBench generalization: residual line overload after voltage-oriented storage dispatch.",
        "claim_cn": "热约束感知修复针对 SimBench 泛化中发现的关键局限：仅电压导向储能调度后仍可能存在残余线路过载。",
        "datasets": "SimBench sample; MIT-derived AI profile",
        "scripts": "code/simbench_thermal_aware_repair.py",
        "evidence_files": "results/00_analysis_outputs/simbench_thermal_aware_8h_summary.csv; results/00_analysis_outputs/SIMBENCH_THERMAL_AWARE_8H_LOG.md",
        "certificate": "pandapower AC power-flow verification with voltage and line-loading metrics",
        "current_status": "supported as 8-hour prototype",
        "next_needed": "Run 12-hour and then full 24-hour thermal-aware repair; compare against voltage-only repair.",
    },
    {
        "claim_id": "C9",
        "claim": "The long-running experimental matrix is complete with 12-hour, 24-hour, and multi-network comparison tables.",
        "claim_cn": "长时实验矩阵已经包含 12 小时、24 小时和多网络对比结果表。",
        "datasets": "Completed batch experiment result tables",
        "scripts": "code/simbench_voltage_thermal_benchmark.py",
        "evidence_files": "results/00_analysis_outputs/results/thermal_aware_12h_summary.csv; results/00_analysis_outputs/results/thermal_aware_24h_summary.csv; results/00_analysis_outputs/results/multinetwork_12h_summary.csv",
        "certificate": "rows with physical_certificate=True, physical_certificate_tool, and final_empirical_claim_allowed=True support final conclusions",
        "current_status": "complete for empirical analysis",
        "next_needed": "Keep provenance fields in final tables.",
    },
    {
        "claim_id": "C10",
        "claim": "The sensitivity-analysis design covers AI-load ratio, storage capacity, interconnection location, and forecast-error/MPC robustness.",
        "claim_cn": "敏感性分析已经覆盖 AI 负荷比例、储能容量、接入节点位置以及预测误差/MPC 鲁棒性。",
        "datasets": "Completed sensitivity-analysis result tables",
        "scripts": "code/simbench_voltage_thermal_benchmark.py",
        "evidence_files": "results/00_analysis_outputs/results/sensitivity_ai_load_ratio.csv; results/00_analysis_outputs/results/sensitivity_storage_capacity.csv; results/00_analysis_outputs/results/sensitivity_interconnection_bus.csv; results/00_analysis_outputs/results/sensitivity_forecast_error_mpc.csv",
        "certificate": "rows with physical_certificate=True, physical_certificate_tool, and final_empirical_claim_allowed=True support final conclusions",
        "current_status": "complete for empirical analysis",
        "next_needed": "Optional future work can add confidence intervals.",
    },
    {
        "claim_id": "C11",
        "claim": "An AC feasibility-ceiling baseline gives a strong reference for judging deployable storage policies.",
        "claim_cn": "AC 可行上限强基线为评价可部署储能策略提供了更强参照。",
        "datasets": "SimBench sample; MIT-derived AI profile",
        "scripts": "code/strong_baseline_ac_feasibility_ceiling.py",
        "evidence_files": "results/00_analysis_outputs/strong_baseline_ac_ceiling_summary.csv; results/00_analysis_outputs/strong_baseline_ac_ceiling_results.csv; results/00_analysis_outputs/STRONG_BASELINE_AC_CEILING_LOG.md",
        "certificate": "pandapower AC power-flow verification for storage-limited and oracle ceiling baselines",
        "current_status": "supported as strong baseline comparison",
        "next_needed": "Report the oracle ceiling as non-deployable and separate it from service-preserving storage policies.",
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "claim_evidence_matrix.csv"
    md_path = OUT / "CLAIM_EVIDENCE_MATRIX.md"

    fields = list(CLAIMS[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(CLAIMS)

    lines = ["# Claim-Evidence Matrix", ""]
    lines.append("| ID | Claim CN | Evidence | Certificate | Status | Next Needed |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in CLAIMS:
        lines.append(
            "| {claim_id} | {claim_cn} | {evidence_files} | {certificate} | {current_status} | {next_needed} |".format(
                **row
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
