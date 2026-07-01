from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT
INDEX = PACKAGE_ROOT / "agent_rag" / "rag_knowledge_base" / "index"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def load_chunks() -> dict[str, dict[str, object]]:
    chunks_path = INDEX / "chunks.jsonl"
    chunks: dict[str, dict[str, object]] = {}
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            chunks[str(item["chunk_id"])] = item
    return chunks


def search(query: str, top_k: int = 5) -> list[tuple[float, dict[str, object]]]:
    chunks = load_chunks()
    index = json.loads((INDEX / "tfidf_index.json").read_text(encoding="utf-8"))
    doc_count = int(index["doc_count"])
    doc_freq = index["doc_freq"]
    norms = index["norms"]
    inverted = index["inverted"]

    q_tf = Counter(tokenize(query))
    q_vec: dict[str, float] = {}
    for term, count in q_tf.items():
        df = int(doc_freq.get(term, 0))
        idf = math.log((1 + doc_count) / (1 + df)) + 1.0
        q_vec[term] = (1.0 + math.log(count)) * idf
    q_norm = math.sqrt(sum(w * w for w in q_vec.values())) or 1.0

    scores: defaultdict[str, float] = defaultdict(float)
    for term, q_weight in q_vec.items():
        for chunk_id, doc_weight in inverted.get(term, []):
            scores[chunk_id] += q_weight * float(doc_weight)

    ranked: list[tuple[float, dict[str, object]]] = []
    for chunk_id, dot in scores.items():
        score = dot / (q_norm * float(norms.get(chunk_id, 1.0)))
        ranked.append((score, chunks[chunk_id]))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the local RAG evidence index.")
    parser.add_argument("query", help="Question or keywords.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a readable summary.")
    args = parser.parse_args()

    results = search(args.query, args.top_k)
    if args.json:
        print(json.dumps([{"score": s, **chunk} for s, chunk in results], ensure_ascii=False, indent=2))
        return

    for rank, (score, chunk) in enumerate(results, 1):
        print(f"[{rank}] score={score:.4f} source={chunk['source']} chunk={chunk['chunk_id']}")
        print(str(chunk["preview"])[:500])
        print()


if __name__ == "__main__":
    main()
