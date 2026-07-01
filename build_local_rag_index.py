from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT
OUT = PACKAGE_ROOT / "results" / "00_analysis_outputs"
KB = PACKAGE_ROOT / "agent_rag" / "rag_knowledge_base"
DOCS = KB / "docs"
INDEX = KB / "index"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def table_as_markdown(rows: list[dict[str, str]], max_rows: int = 20) -> str:
    if not rows:
        return "No rows available.\n"
    fields = list(rows[0].keys())
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def write_doc(name: str, title: str, body: str) -> Path:
    path = DOCS / name
    path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")
    return path


def generate_knowledge_docs() -> list[Path]:
    DOCS.mkdir(parents=True, exist_ok=True)
    INDEX.mkdir(parents=True, exist_ok=True)

    dataset_inventory = read_csv_rows(OUT / "dataset_inventory.csv")
    mit_metrics = read_csv_rows(OUT / "mit_gpu_trace_metrics.csv")
    workload_metrics = read_csv_rows(OUT / "ai_workload_scenario_library_metrics.csv")
    moderate_summary = read_csv_rows(OUT / "ieee33_ai_load_powerflow_summary_moderate.csv")
    optimized_summary = read_csv_rows(OUT / "ieee33_ai_load_powerflow_summary_optimized_storage.csv")
    iterative_summary = read_csv_rows(OUT / "ieee33_iterative_physics_certified_summary.csv")
    simbench_summary = read_csv_rows(OUT / "simbench_ai_generalization_summary.csv")
    repair_iterations = read_csv_rows(OUT / "iterative_repair_iteration_summary.csv")
    agent_eval_summary = read_csv_rows(OUT / "agent_rag_repair_memory_evaluation_summary.csv")
    thermal_summary = read_csv_rows(OUT / "simbench_thermal_aware_8h_summary.csv")
    completed_baseline_rank = read_csv_rows(OUT / "results" / "baseline_comparison_rank.csv")
    sensitivity_aggregate = read_csv_rows(OUT / "results" / "sensitivity_all_aggregate.csv")
    strong_baseline_summary = read_csv_rows(OUT / "strong_baseline_ac_ceiling_summary.csv")

    docs: list[Path] = []

    docs.append(
        write_doc(
            "01_problem_and_claims.md",
            "Problem Definition and Research Claims",
            """
Research identity: Physics-Certified Multi-Agent RAG for Grid-Interactive AI Data Centers under LLM-Induced Load Transients.

Core power-system problem: AI/HPC workloads introduce high-ramp, high-variability electric demand. The system goal is to transform real AI workload traces into grid-seen loads that reduce voltage and thermal violations while preserving service demand.

high-standard power-system positioning: the LLM/Agent layer is an orchestration and evidence-retrieval layer. Numerical validity must come from deterministic optimization and AC power-flow verification, not free-form language generation.

Primary claim to test: a storage/UPS dispatch schedule that is optimized and then repaired using AC power-flow counterexamples can reduce feeder stress relative to raw AI load and rule-based smoothing.

Recommended breakthrough: make the Agent/RAG innovation auditable. The agent retrieves datasets, formulas, constraints, prior failed cases, and tool results; it proposes a tool plan; physical tools issue certificates; failed certificates drive repair.

中文检索关键词：研究问题，突破口，高水平创新点，AI数据中心，LLM负荷，电力系统，大语言模型，智能体，物理证书，确定性验证，配电网，电压越限，线路过载，储能调度，UPS调度，Agent RAG。
""",
        )
    )

    docs.append(
        write_doc(
            "02_datasets.md",
            "Datasets and Traceability",
            f"""
The current local data package is stored under `{ROOT}`.

Dataset inventory:

{table_as_markdown(dataset_inventory)}

MIT Supercloud GPU trace metrics:

{table_as_markdown(mit_metrics)}

The MIT Supercloud trace is used to construct AI workload scenarios. IEEE 33-bus is used for the first controlled distribution-feeder study. SimBench is used for multi-topology generalization. OPSD can provide background time-series context. PGLib can later support OPF-style comparisons.

中文检索关键词：AI_DataCenter_AgentRAG_PowerSystem_Research_Package，MIT Supercloud，IEEE 33节点，IEEE 33-bus，SimBench，pandapower，PGLib，Open Power System Data，OPSD，负荷曲线，GPU功率轨迹，数据溯源。
""",
        )
    )

    docs.append(
        write_doc(
            "03_ai_workload_library.md",
            "AI Workload Scenario Library",
            f"""
Four AI workload classes were generated from the representative MIT Supercloud trace:

{table_as_markdown(workload_metrics)}

Interpretation:

- training_sustained is near-flat and is useful as a low-ramp baseline.
- finetune_stagewise captures staged load transitions.
- inference_burst captures high peak-to-mean and high burstiness.
- rag_batch represents retrieval/batch serving cycles and is directly relevant to Agent/RAG infrastructure.

These classes should be treated as controllable scenarios, not as universal empirical laws. In a technical venue study, the contribution is the trace-to-grid scenario construction and the certified dispatch workflow.

中文检索关键词：AI负荷场景库，训练负荷，微调负荷，推理突发，RAG批处理，峰均比，爬坡率，负荷波动，LLM推理，GPU负荷。
""",
        )
    )

    docs.append(
        write_doc(
            "04_physics_certified_method.md",
            "Physics-Certified Dispatch Method",
            """
Raw AI load is transformed into grid-seen AI load by local storage or UPS:

P_ai_grid(t) = P_ai_raw(t) + P_ch(t) - P_dis(t)

Storage dynamics:

SOC(t+1) = SOC(t) + eta_ch * P_ch(t) * Delta_t - P_dis(t) * Delta_t / eta_dis

Core constraints:

- 0 <= P_ch(t) <= P_ch_max
- 0 <= P_dis(t) <= P_dis_max
- SOC_min <= SOC(t) <= SOC_max
- SOC(T) = SOC(0)
- grid-seen peak and ramp are penalized or bounded

The key methodological distinction is the certificate loop:

1. Optimize a dispatch using a tractable proxy.
2. Verify every hour with AC power flow in pandapower.
3. Record counterexamples such as low voltage, overload, or non-convergence.
4. Update stress weights or AI-load ceilings.
5. Re-optimize and verify again.

This produces an optimization-verification-repair loop. The Agent may choose which deterministic tool to run, but it cannot certify physics by itself.

中文检索关键词：物理认证，物理证书，优化-验证-修复闭环，潮流计算，AC潮流，pandapower，电压约束，SOC约束，爬坡约束，峰值约束，反例修复。
""",
        )
    )

    docs.append(
        write_doc(
            "05_ieee33_results.md",
            "IEEE 33-Bus Experimental Evidence",
            f"""
Moderate raw and smoothed AI-load summary:

{table_as_markdown(moderate_summary)}

One-shot optimized storage summary:

{table_as_markdown(optimized_summary)}

Iterative physics-certified summary:

{table_as_markdown(iterative_summary)}

Current interpretation:

- Raw AI load creates voltage stress, especially when injected at electrically weak buses.
- Rule-based smoothing alone does not remove most voltage violations.
- Storage optimization reduces selected metrics but can still miss AC power-flow constraints.
- Iterative repair is more defensible for a high-standard power-system study because it explicitly uses failed physical certificates as feedback.

中文检索关键词：IEEE 33节点实验，原始AI负荷，平滑负荷，储能优化，迭代物理认证，电压越限小时数，欠压，网损，薄弱节点，18号节点，33号节点。
""",
        )
    )

    docs.append(
        write_doc(
            "06_simbench_generalization.md",
            "SimBench Generalization Evidence",
            f"""
SimBench multi-topology summary:

{table_as_markdown(simbench_summary)}

Current interpretation:

The SimBench experiment is not yet the final large-scale benchmark, but it already supports the generalization narrative: the method can be moved from IEEE 33-bus to realistic European-style benchmark feeders, and the same certificate metrics can be recorded across topologies.

For the next high-standard step, increase the number of representative hours, add more SimBench networks, and report both violation reduction and computational cost.

中文检索关键词：SimBench泛化，多拓扑验证，欧洲配电网，代表小时，线路过载，泛化实验，跨网络验证，计算成本，高标准实验。
""",
        )
    )

    docs.append(
        write_doc(
            "07_agent_rag_design.md",
            "Agent and RAG Design",
            """
Recommended model roles:

- Primary planning and research-writing Agent: gpt-5.5 through the Responses API, reasoning effort medium or high depending on latency and cost.
- Retrieval embeddings: text-embedding-3-small for low-cost local document indexing, or text-embedding-3-large when recall quality is more important than cost.
- Lightweight router or formatter: a small model can be used later, but it should not make numerical power-system decisions.

Recommended agents:

- RequestAgent: convert a natural-language research or operation request into a typed task.
- RetrievalAgent: retrieve datasets, formulas, prior experiment results, constraints, and failed certificate records.
- FormulationAgent: assemble an optimization/tool-call plan.
- VerificationAgent: run deterministic tools and issue certificates.
- RepairAgent: use failed certificates to adjust stress weights, safe ceilings, or scenario assumptions.

RAG content must include experiment outputs, data lineage, mathematical constraints, failure logs, and figure references. It should not be only text abstracts.

中文检索关键词：Agent模型选择，RAG构建，Responses API，gpt-5.5，text-embedding-3-small，text-embedding-3-large，RequestAgent，RetrievalAgent，FormulationAgent，VerificationAgent，RepairAgent，工具调用。
""",
        )
    )

    docs.append(
        write_doc(
            "08_repair_memory.md",
            "Repair Memory and Failure Cases",
            f"""
Iterative repair records:

{table_as_markdown(repair_iterations, max_rows=30)}

The repair memory is important for the Agent contribution. It gives the Agent a grounded memory of what failed, when it failed, and which repair was attempted. In a final system, this memory can be queried before running new optimization cases.

中文检索关键词：修复记忆，失败案例，失败证书，反例，越限小时，修复动作，安全上限，压力权重，迭代次数，证书通过率。
""",
        )
    )

    docs.append(
        write_doc(
            "09_agent_rag_evaluation.md",
            "Agent/RAG Repair-Memory Evaluation",
            f"""
Agent/RAG repair-memory evaluation summary:

{table_as_markdown(agent_eval_summary)}

Interpretation:

The current Agent/RAG prototype should be claimed as an orchestration and traceability improvement, not as an autonomous physical solver. In the current comparison, the RAG-memory policy selects the iterative repair tool and retrieves required evidence sources. The physical certificate metrics are close to the no-RAG one-shot strategy, with small undervoltage-bus-hour reduction but no strong voltage-margin improvement yet.

This is useful for a high-standard power-system path because it prevents overclaiming. The next experiment should strengthen the repair algorithm and report Agent metrics separately from power-system metrics.

中文检索关键词：Agent消融，RAG修复记忆，工具计划可执行率，证据完整率，证书通过率，no-RAG，一阶段优化，物理收益，过度宣称，高水平研究。
""",
        )
    )

    docs.append(
        write_doc(
            "10_thermal_aware_repair.md",
            "Thermal-Aware SimBench Repair Prototype",
            f"""
Thermal-aware SimBench repair summary:

{table_as_markdown(thermal_summary)}

Interpretation:

The 12-hour SimBench extension showed that voltage-oriented storage dispatch can reduce undervoltage but may leave thermal overload risk. This prototype adds line-loading-aware stress and upper-bound repair. In the 8-hour prototype, thermal-aware storage reduces overloaded line-hours and also improves undervoltage counts relative to raw AI load.

This strengthens the algorithmic direction: the repair loop should become voltage-plus-thermal certified, and later can be upgraded to explicit OPF-style constraints or line-loading cuts.

中文检索关键词：热稳定，线路过载，thermal-aware，线路载流率，过载小时，SimBench修复，电压加热约束，OPF约束，线路约束，物理认证。
""",
        )
    )

    docs.append(
        write_doc(
            "11_completed_batch_results.md",
            "Completed Batch Experiment Results",
            f"""
Important: this document describes certificate-tagged batch experiment results
generated by the local benchmark workflow. Rows with `physical_certificate=True`,
an explicit `physical_certificate_tool`, and `final_empirical_claim_allowed=True`
are valid evidence for final empirical conclusions when the manuscript cites the
exact source table and generation script.

Completed baseline ranking:

{table_as_markdown(completed_baseline_rank)}

The batch experiment package covers:

- thermal-aware repair extended to 12 representative hours;
- thermal-aware repair extended to 24 hours;
- multi-network SimBench generalization on six candidate networks;
- stronger baseline comparison including raw AI, moving-average smoothing, rule-based peak shaving, voltage-only storage, and voltage+thermal storage.

Research implication:

The batch trend is consistent with the 8-hour thermal-aware prototype:
voltage+thermal repair outperforms voltage-only repair primarily in overloaded
line-hours, while maintaining or improving undervoltage performance. The result
should be reported as a certificate-backed batch experiment, not as an LLM-made
claim.

中文检索关键词：完成结果，结果表，实验完整性，12小时，24小时，多网络，强基线，moving average，peak shaving，voltage-only，thermal-aware，可作为已完成实验结果使用。
""",
        )
    )

    docs.append(
        write_doc(
            "12_sensitivity_analysis.md",
            "Sensitivity Analysis",
            f"""
Important: this document describes certificate-tagged sensitivity results from
the local batch workflow. Rows with `physical_certificate=True`, an explicit
`physical_certificate_tool`, and `final_empirical_claim_allowed=True` are valid
evidence for final sensitivity conclusions when the manuscript maps each claim
to the exact source CSV and generation script.

Sensitivity aggregate table:

{table_as_markdown(sensitivity_aggregate, max_rows=80)}

The completed sensitivity package covers:

- AI-load ratio: 0.5%, 1.0%, 1.5%, 3.0%, and 5.0% of base load;
- storage capacity multiplier: 0, 0.5, 1.0, 2.0, and 4.0;
- interconnection bus location: upstream, mid-feeder, branch, lateral end, and feeder end;
- forecast error and MPC: day-ahead dispatch, receding-horizon MPC, and robust MPC.

Expected trends:

- higher AI-load ratio increases undervoltage and line-overload risk;
- larger storage capacity reduces violations with diminishing returns;
- weaker downstream buses are more vulnerable;
- MPC is expected to degrade less than day-ahead dispatch under forecast error;
- voltage+thermal-aware storage should outperform voltage-only storage mainly on line-loading metrics.

中文检索关键词：敏感性分析，AI负荷比例，储能容量，接入节点，预测误差，MPC，鲁棒MPC，容量倍数，负荷比例，弱节点，完成结果。
""",
        )
    )

    docs.append(
        write_doc(
            "13_strong_baseline_ac_ceiling.md",
            "Strong Baseline: AC Feasibility Ceiling",
            f"""
This document records the added strong baseline comparison. The baseline is
generated by `code/strong_baseline_ac_feasibility_ceiling.py` and verified with
pandapower AC power flow.

Strong-baseline summary:

{table_as_markdown(strong_baseline_summary)}

Interpretation:

- `ac_feasibility_ceiling_storage` uses the same service-preserving storage
  assumptions and should be compared with deployable storage policies.
- `ac_feasibility_ceiling_oracle` is a non-deployable AC-feasible ceiling. It is
  useful as a theoretical reference, but it should not be described as an
  implementable controller.
- Numerical conclusions from this baseline are acceptable only when tied to the
  generated CSV outputs and the pandapower certificate metrics.

中文检索关键词：强基线，AC 可行上限，oracle ceiling，storage-limited baseline，pandapower 证书，SimBench，对比实验，可部署策略，非可部署理论上限。
""",
        )
    )

    return docs


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def chunk_markdown(path: Path, max_tokens: int = 230, overlap: int = 40) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    tokens = tokenize(text)
    chunks: list[dict[str, object]] = []
    if not tokens:
        return chunks
    step = max(1, max_tokens - overlap)
    for start in range(0, len(tokens), step):
        end = min(len(tokens), start + max_tokens)
        token_slice = tokens[start:end]
        preview = " ".join(token_slice[:80])
        chunks.append(
            {
                "chunk_id": f"{path.stem}:{len(chunks):03d}",
                "source": str(path.relative_to(KB)).replace("\\", "/"),
                "title": path.stem,
                "token_start": start,
                "token_end": end,
                "text": " ".join(token_slice),
                "preview": preview,
            }
        )
        if end == len(tokens):
            break
    return chunks


def build_tfidf(chunks: list[dict[str, object]]) -> dict[str, object]:
    doc_freq: Counter[str] = Counter()
    term_freqs: dict[str, Counter[str]] = {}
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        tf = Counter(str(chunk["text"]).split())
        term_freqs[chunk_id] = tf
        doc_freq.update(tf.keys())

    n_docs = max(1, len(chunks))
    vectors: dict[str, dict[str, float]] = {}
    norms: dict[str, float] = {}
    inverted: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for chunk_id, tf in term_freqs.items():
        vec: dict[str, float] = {}
        for term, count in tf.items():
            idf = math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0
            weight = (1.0 + math.log(count)) * idf
            vec[term] = weight
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        norms[chunk_id] = norm
        vectors[chunk_id] = vec
        for term, weight in vec.items():
            inverted[term].append((chunk_id, weight))

    return {
        "doc_count": n_docs,
        "doc_freq": dict(doc_freq),
        "norms": norms,
        "inverted": {term: postings for term, postings in inverted.items()},
    }


def main() -> None:
    docs = generate_knowledge_docs()
    chunks: list[dict[str, object]] = []
    for doc in docs:
        chunks.extend(chunk_markdown(doc))

    index = build_tfidf(chunks)

    chunks_path = INDEX / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    index_path = INDEX / "tfidf_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "retrieval": "local_tfidf",
        "optional_embedding_model": "text-embedding-3-small or text-embedding-3-large",
        "primary_agent_model": "gpt-5.5",
        "principle": "RAG retrieves evidence; deterministic tools certify physics.",
        "docs": [str(doc.relative_to(KB)).replace("\\", "/") for doc in docs],
    }
    (INDEX / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
