#!/usr/bin/env python3
"""Lab 3 — retrieval sweeps.

The scaffolding (corpus loading, metric computation, table printing) is
written for you. The sweeps are yours.

    python labs/lab3/search.py --baseline
    python labs/lab3/search.py --sweep chunking
    python labs/lab3/search.py --sweep retrieval
    python labs/lab3/search.py --sweep rerank
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import STRATEGIES, Chunk  # noqa: E402
from aip.evals import retrieval_metrics  # noqa: E402
from aip.retrieval import (
    Bm25Retriever,
    DenseRetriever,
    HybridRetriever,
    Retriever,
    CrossEncoderReranker,
    LLMReranker,
    ChromaRetriever,
)

CORPUS_DIR = ROOT / "data/corpus"
GOLDEN = ROOT / "data/eval/rag_golden.jsonl"


# ---------------------------------------------------------------------------
# scaffolding (provided)
# ---------------------------------------------------------------------------
def load_corpus() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS_DIR.glob("*.md"))}


def load_questions(include_unanswerable: bool = False) -> list[dict]:
    rows = [json.loads(l) for l in GOLDEN.open(encoding="utf-8")]
    if include_unanswerable:
        return rows
    # THREE questions (Q36, Q38, Q39) have no relevant document, so recall and
    # nDCG are undefined for them -- you cannot rank correctly against an empty
    # relevant set. Dropping them leaves n = 42.
    #
    # Do not confuse that with the FIVE questions of kind 'unanswerable'
    # (Q36-Q40): two of those do keep relevant documents, because part of what
    # they ask is supported. All five are measured properly in Lab 4, as
    # refusal precision and recall.
    #
    # Excluding the three is correct -- but say so in your report rather than
    # letting an unexplained n = 42 pass for a stated 45.
    return [r for r in rows if r["relevant_docs"]]


def build_chunks(corpus: dict[str, str], strategy: str = "sliding",
                 size: int = 800, **kw) -> list[Chunk]:
    fn = STRATEGIES[strategy]
    out: list[Chunk] = []
    for doc_id, text in corpus.items():
        try:
            out.extend(fn(text, doc_id, size=size, **kw))
        except TypeError:                       # chunker without that kwarg
            out.extend(fn(text, doc_id, size=size))
    return out


def evaluate(retriever: Retriever, questions: list[dict], k: int = 10,
             reranker=None, final_k: int = 5) -> dict:
    """Run every question, return aggregate metrics + per-kind breakdown."""
    agg: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latencies: list[float] = []
    per_q: dict[str, float] = {}
    per_q_mrr: dict[str, float] = {}

    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k=k)
        if reranker is not None:
            hits = reranker.rerank(q["question"], hits, k=final_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        # A document counts as retrieved at rank r if any of its chunks does.
        seen, ranked = set(), []
        for h in hits:
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                ranked.append(h.doc_id)

        m = retrieval_metrics(ranked, q["relevant_docs"], ks=(1, 3, 5, 10))
        per_q[q["id"]] = m["hit_rate@5"]
        per_q_mrr[q["id"]] = m["mrr"]
        for key, val in m.items():
            agg[key].append(val)
            by_kind[q["kind"]][key].append(val)

    out = {k2: statistics.fmean(v) for k2, v in agg.items()}
    out["latency_p50_ms"] = statistics.median(latencies)
    out["latency_p95_ms"] = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    out["_by_kind"] = {kind: {k2: statistics.fmean(v) for k2, v in d.items()}
                       for kind, d in by_kind.items()}
    out["_per_question"] = per_q            # hit_rate@5 -- saturated, see kind_table
    out["_per_question_mrr"] = per_q_mrr    # use this one for Part B
    out["_kind_n"] = {kind: len(d["mrr"]) for kind, d in by_kind.items()}
    return out


def table(rows: dict[str, dict], cols: tuple[str, ...] =
          ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10",
           "latency_p95_ms")) -> str:
    name_w = max(len(n) for n in rows) + 2
    head = f"{'config':<{name_w}}" + "".join(f"{c:>15}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, m in rows.items():
        lines.append(f"{name:<{name_w}}" + "".join(f"{m.get(c, 0):>15.4f}" for c in cols))
    return "\n".join(lines)


def kind_table(metrics: dict, col: str = "hit_rate@5") -> str:
    """Break a result down by question kind.

    NOTE the default column. `hit_rate@5` is saturated on this corpus -- every
    retriever scores 0.93-0.98 -- so this table will look flat and tell you
    nothing. Pass col='mrr' or col='ndcg@10' for Part B. The default is left
    saturated on purpose.
    """
    bk, counts = metrics["_by_kind"], metrics.get("_kind_n", {})
    w = max(len(k) for k in bk) + 2
    lines = [f"{'kind':<{w}}{col:>12}{'n':>6}", "-" * (w + 18)]
    for kind, m in sorted(bk.items()):
        lines.append(f"{kind:<{w}}{m.get(col, 0):>12.4f}{counts.get(kind, 0):>6}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# sweeps (yours)
# ---------------------------------------------------------------------------
def sweep_baseline() -> None:
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "sliding", 800, overlap=150)
    print(f"corpus: {len(corpus)} docs -> {len(chunks)} chunks "
          f"(mean {statistics.fmean(len(c) for c in chunks):.0f} chars)")
    r = DenseRetriever(chunks)
    m = evaluate(r, questions)
    print(table({"baseline sliding-800 dense": m}))
    print()
    print(kind_table(m))
    print("\nWrite these numbers down before you change anything.")


def sweep_chunking() -> None:
    """A1-A3: chunking strategy, size, and heading-prefix sweeps."""
    corpus, questions = load_corpus(), load_questions()

    def run(chunks: list) -> dict:
        t0 = time.perf_counter()
        r = DenseRetriever(chunks)
        build_s = time.perf_counter() - t0
        m = evaluate(r, questions)
        m["_n_chunks"] = len(chunks)
        m["_build_s"] = build_s
        return m

    def strip_prefix(text: str) -> str:
        if text.startswith("[") and "]\n" in text:
            return text.split("]\n", 1)[1]
        return text

    # --- A1: all four strategies at size=800 ---
    a1_results = {}
    for strategy in STRATEGIES:
        chunks = build_chunks(corpus, strategy, size=800)
        a1_results[strategy] = run(chunks)

    print("=== A1: chunking strategy @ size=800 ===")
    print(table(a1_results))
    print("chunk counts:", {k: v["_n_chunks"] for k, v in a1_results.items()})
    print("build time (s):", {k: round(v["_build_s"], 2) for k, v in a1_results.items()})

    winner = max(a1_results, key=lambda k: a1_results[k]["ndcg@10"])
    print(f"\nwinner: {winner}")

    # --- A2: sweep size on the winner ---
    a2_results = {}
    for size in (400, 800, 1600):
        chunks = build_chunks(corpus, winner, size=size)
        a2_results[f"{winner}-{size}"] = run(chunks)

    print("\n=== A2: size sweep on winner ===")
    print(table(a2_results))
    print("chunk counts:", {k: v["_n_chunks"] for k, v in a2_results.items()})

    # --- A3: markdown heading-prefix ablation ---
    md_chunks = build_chunks(corpus, "markdown", size=800)
    stripped = [
        Chunk(strip_prefix(c.text), c.doc_id, c.chunk_id, c.meta)
        for c in md_chunks
    ]
    a3_results = {
        "markdown-with-prefix": run(md_chunks),
        "markdown-no-prefix": run(stripped),
    }
    print("\n=== A3: heading-path prefix ablation ===")
    print(table(a3_results))
    delta = a3_results["markdown-with-prefix"]["ndcg@10"] - a3_results["markdown-no-prefix"]["ndcg@10"]
    print(f"nDCG@10 delta from prefix: {delta:+.4f}")


def sweep_retrieval() -> None:
    """B1-B4: dense vs BM25 vs hybrid, broken down by question kind."""
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "markdown", size=400)  # your A1/A2 winner

    dense = DenseRetriever(chunks)
    bm25 = Bm25Retriever(chunks)
    hybrid = HybridRetriever([dense, bm25])

    results = {"dense": evaluate(dense, questions),
               "bm25": evaluate(bm25, questions),
               "hybrid": evaluate(hybrid, questions)}

    print("=== B1: dense vs bm25 vs hybrid ===")
    print(table(results))

    print("\n=== B2: per-kind breakdown (MRR, not hit_rate@5) ===")
    for name, m in results.items():
        print(f"\n--- {name} ---")
        print(kind_table(m, col="mrr"))

    print("\n=== B2b: Q44 and Q41 individually ===")
    for name, m in results.items():
        q44 = m["_per_question_mrr"].get("Q44")
        q41 = m["_per_question_mrr"].get("Q41")
        print(f"{name}: Q44={q44}, Q41={q41}")

    print("\n=== B3: RRF k sweep ===")
    for k in (10, 30, 60, 100):
        h = HybridRetriever([dense, bm25], rrf_k=k)
        m = evaluate(h, questions)
        print(f"k={k}: ndcg@10={m['ndcg@10']:.4f}")

    print("\n=== B4: unequal fusion weights ===")
    for w in ([1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [3.0, 1.0]):
        h = HybridRetriever([dense, bm25], weights=w)
        m = evaluate(h, questions)
        print(f"weights={w}: ndcg@10={m['ndcg@10']:.4f}")


def sweep_rerank() -> None:
    """C1-C4: cross-encoder vs LLM reranker, and the deployment decision."""
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "markdown", size=400)  # your winner
    dense = DenseRetriever(chunks)

    baseline = evaluate(dense, questions, k=10)
    print("=== baseline (no rerank, k=10) ===")
    print(table({"dense": baseline}))

    ce = CrossEncoderReranker()
    ce_result = evaluate(dense, questions, k=30, reranker=ce, final_k=5)
    print("\n=== C1: cross-encoder rerank (k=30 -> 5) ===")
    print(table({"cross-encoder": ce_result}, cols=(
        "hit_rate@1", "recall@5", "ndcg@10", "latency_p95_ms")))

    llm = LLMReranker()
    t0 = time.perf_counter()
    llm_result = evaluate(dense, questions, k=30, reranker=llm, final_k=5)
    llm_wall_s = time.perf_counter() - t0
    print("\n=== C2: LLM rerank (k=30 -> 5) ===")
    print(table({"llm": llm_result}, cols=(
        "hit_rate@1", "recall@5", "ndcg@10", "latency_p95_ms")))
    print(f"wall time for full sweep: {llm_wall_s:.1f}s over {len(questions)} questions")

    # C3: decision table
    print("\n=== C3: decision table ===")
    print(f"{'config':<16}{'ndcg@10':>10}{'hit@1':>10}{'p95_ms':>10}")
    for name, m in [("none", baseline), ("cross-encoder", ce_result), ("llm", llm_result)]:
        print(f"{name:<16}{m['ndcg@10']:>10.4f}{m['hit_rate@1']:>10.4f}{m['latency_p95_ms']:>10.1f}")

    # C4: find a regression
    print("\n=== C4: questions where cross-encoder rerank hurt MRR ===")
    for qid, base_mrr in baseline["_per_question_mrr"].items():
        rerank_mrr = ce_result["_per_question_mrr"].get(qid)
        if rerank_mrr is not None and rerank_mrr < base_mrr:
            print(f"{qid}: {base_mrr:.2f} -> {rerank_mrr:.2f}")


def sweep_index() -> None:
    """D1-D3: Index comparison, scaling, and metadata filtering."""

    import time
    from pathlib import Path

    corpus, questions = load_corpus(), load_questions()

    # ============================================================
    # Best configuration from Part A
    # Markdown chunking, size 400, heading path enabled
    # ============================================================
    chunks = build_chunks(
        corpus,
        "markdown",
        400,
        include_heading_path=True
    )

    # ============================================================
    # D1: Exact NumPy vs Chroma HNSW
    # ============================================================
    print("\nD1: Exact NumPy vs Chroma HNSW")

    # ------------------------------------------------------------
    # Exact NumPy dense retrieval
    # ------------------------------------------------------------
    dense = DenseRetriever(chunks)

    dense_metrics = evaluate(
        dense,
        questions
    )

    # ------------------------------------------------------------
    # Chroma HNSW
    # ------------------------------------------------------------
    chroma = ChromaRetriever(
        chunks,
        path=".chroma_lab3",
        collection="lab3_d1",
        reset=True
    )

    chroma_metrics = evaluate(
        chroma,
        questions
    )

    print("\nExact NumPy")
    print(table({
        "dense-exact": dense_metrics
    }))

    print("\nChroma HNSW")
    print(table({
        "chroma-hnsw": chroma_metrics
    }))

    print(
        f"\nRecall@5 gap: "
        f"{chroma_metrics['recall@5'] - dense_metrics['recall@5']:+.4f}"
    )

    print(
        f"nDCG@10 gap: "
        f"{chroma_metrics['ndcg@10'] - dense_metrics['ndcg@10']:+.4f}"
    )

    # ============================================================
    # D2: Scaling experiment
    # ============================================================
    print("\nD2: Scaling experiment")

    root = Path(__file__).resolve().parents[2]
    scaled_dir = root / "data" / "corpus_scaled"

    # Use one fixed query for a fair latency comparison.
    timing_query = questions[0]["question"]

    def time_retriever(
        retriever,
        query: str,
        repeats: int = 5
    ) -> float:
        """Return median search latency in milliseconds."""

        times = []

        # Warm-up search
        retriever.search(query, k=10)

        # Timed searches
        for _ in range(repeats):
            start = time.perf_counter()

            retriever.search(
                query,
                k=10
            )

            elapsed = (
                time.perf_counter() - start
            ) * 1000

            times.append(elapsed)

        times.sort()

        # Median
        return times[len(times) // 2]

    def run_scale(
        label: str,
        corpus_for_index: dict
    ):
        """Build exact and HNSW indexes and compare latency."""

        # Build chunks
        scale_chunks = build_chunks(
            corpus_for_index,
            "markdown",
            400,
            include_heading_path=True
        )

        print(f"\n{label}")
        print(f"Chunks: {len(scale_chunks)}")

        # --------------------------------------------------------
        # Exact NumPy
        # --------------------------------------------------------
        exact = DenseRetriever(
            scale_chunks
        )

        exact_ms = time_retriever(
            exact,
            timing_query
        )

        # --------------------------------------------------------
        # Chroma HNSW
        # --------------------------------------------------------
        collection_name = (
            "lab3_scale_"
            + label.lower()
            .replace(" ", "_")
            .replace("~", "")
        )

        hnsw = ChromaRetriever(
            scale_chunks,
            path=".chroma_lab3",
            collection=collection_name,
            reset=True
        )

        hnsw_ms = time_retriever(
            hnsw,
            timing_query
        )

        ratio = (
            hnsw_ms / exact_ms
            if exact_ms
            else float("inf")
        )

        print(
            f"Exact NumPy: {exact_ms:.3f} ms"
        )

        print(
            f"Chroma HNSW: {hnsw_ms:.3f} ms"
        )

        print(
            f"HNSW / Exact: {ratio:.2f}x"
        )

        return (
            len(scale_chunks),
            exact_ms,
            hnsw_ms
        )

    # Store D2 measurements
    d2_results = []

    # ------------------------------------------------------------
    # D2.1: Original corpus
    # Approximately 160 chunks
    # ------------------------------------------------------------
    d2_results.append(
        run_scale(
            "~160 chunks",
            corpus
        )
    )

    # ------------------------------------------------------------
    # D2.2 and D2.3 require scaled corpus
    # ------------------------------------------------------------
    if not scaled_dir.exists():

        print(
            "\nScaled corpus not found."
        )

        print(
            "Run:"
        )

        print(
            "python scripts/expand_corpus.py --docs 4000"
        )

    else:

        scaled_files = sorted(
            scaled_dir.glob("*.md")
        )

        # --------------------------------------------------------
        # D2.2: Approximately 4k chunks
        #
        # Each filler document produces roughly 10 chunks.
        # 400 filler documents therefore give approximately 4k
        # chunks.
        # --------------------------------------------------------
        small_scaled_files = scaled_files[:400]

        small_corpus = dict(corpus)

        for p in small_scaled_files:

            small_corpus[p.stem] = (
                p.read_text(
                    encoding="utf-8"
                )
            )

        d2_results.append(
            run_scale(
                "~4k chunks",
                small_corpus
            )
        )

        # --------------------------------------------------------
        # D2.3: Approximately 40k chunks
        #
        # Use all 4,000 filler documents.
        # --------------------------------------------------------
        full_scaled_corpus = dict(corpus)

        for p in scaled_files:

            full_scaled_corpus[p.stem] = (
                p.read_text(
                    encoding="utf-8"
                )
            )

        d2_results.append(
            run_scale(
                "~40k chunks",
                full_scaled_corpus
            )
        )

    # ------------------------------------------------------------
    # D2 Summary
    # ------------------------------------------------------------
    print("\nD2 Summary")

    print(
        f"{'Scale':<15}"
        f"{'Chunks':>10}"
        f"{'Exact ms':>15}"
        f"{'HNSW ms':>15}"
        f"{'HNSW/Exact':>15}"
    )

    print("-" * 70)

    for chunks_count, exact_ms, hnsw_ms in d2_results:

        print(
            f"{'':<15}"
            f"{chunks_count:>10}"
            f"{exact_ms:>15.3f}"
            f"{hnsw_ms:>15.3f}"
            f"{hnsw_ms / exact_ms:>14.2f}x"
        )

    # ============================================================
    # D3: Metadata filtering
    # ============================================================
    print("\nD3: Metadata filtering")

    # ------------------------------------------------------------
    # Add status metadata to every chunk
    #
    # Archived documents have "ARCHIVED" in their doc_id.
    # Everything else is treated as current.
    # ------------------------------------------------------------
    for chunk in chunks:

        chunk.meta["status"] = (
            "archived"
            if "ARCHIVED" in chunk.doc_id
            else "current"
        )

    # ------------------------------------------------------------
    # Fresh Chroma collection so metadata is ingested
    # ------------------------------------------------------------
    chroma_meta = ChromaRetriever(
        chunks,
        path=".chroma_lab3",
        collection="lab3_metadata",
        reset=True
    )

    # ------------------------------------------------------------
    # Questions required by the lab
    # ------------------------------------------------------------
    target_ids = {
        "Q29",
        "Q30",
        "Q31"
    }

    target_questions = [
        q
        for q in questions
        if q["id"] in target_ids
    ]

    # ------------------------------------------------------------
    # D3 Before filtering
    # ------------------------------------------------------------
    before = {}

    for q in target_questions:

        hits = chroma_meta.search(
            q["question"],
            k=10
        )

        ranked = []
        seen = set()

        for h in hits:

            if h.doc_id not in seen:

                seen.add(h.doc_id)
                ranked.append(h.doc_id)

        m = retrieval_metrics(
            ranked,
            q["relevant_docs"],
            ks=(1, 5, 10)
        )

        before[q["id"]] = (
            m["hit_rate@1"]
        )

    # ------------------------------------------------------------
    # D3 After filtering
    #
    # Only current policies are allowed.
    # ------------------------------------------------------------
    after = {}

    for q in target_questions:

        hits = chroma_meta.search(
            q["question"],
            k=10,
            where={
                "status": "current"
            }
        )

        ranked = []
        seen = set()

        for h in hits:

            if h.doc_id not in seen:

                seen.add(h.doc_id)
                ranked.append(h.doc_id)

        m = retrieval_metrics(
            ranked,
            q["relevant_docs"],
            ks=(1, 5, 10)
        )

        after[q["id"]] = (
            m["hit_rate@1"]
        )

    # ------------------------------------------------------------
    # Print D3 results
    # ------------------------------------------------------------
    print(
        "\nQ29-Q31 Hit@1 before vs after metadata filtering"
    )

    print(
        f"{'Question':<15}"
        f"{'Before':>12}"
        f"{'After':>12}"
        f"{'Delta':>12}"
    )

    print("-" * 49)

    for qid in [
        "Q29",
        "Q30",
        "Q31"
    ]:

        delta = (
            after[qid]
            - before[qid]
        )

        print(
            f"{qid:<15}"
            f"{before[qid]:>12.4f}"
            f"{after[qid]:>12.4f}"
            f"{delta:>12.4f}"
        )


SWEEPS = {
    "chunking": sweep_chunking,
    "retrieval": sweep_retrieval,
    "rerank": sweep_rerank,
    "index": sweep_index,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--sweep", choices=list(SWEEPS))
    args = ap.parse_args()
    if args.baseline or not args.sweep:
        sweep_baseline()
    if args.sweep:
        SWEEPS[args.sweep]()


if __name__ == "__main__":
    main()
