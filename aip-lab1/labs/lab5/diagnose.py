#!/usr/bin/env python3
"""Lab 5 — the failure classifier.

    python labs/lab5/diagnose.py --input reports/lab4.json
    python labs/lab5/diagnose.py --input reports/lab4.json --pareto

Implements the T4 §5 diagnostic tree. Everything decidable by code is decided
by code; mode 2 needs your eyes and the script says so.

Note on this pipeline: answer_question() (labs/lab4/rag.py) retrieves k=12,
slices to final_k=5, no reranker. Mode 5 is therefore structurally
unreachable. Also: in_top_30() alone is NOT sufficient to call mode 4 --
a fresh top-30 search can rank the gold doc #1 while it was ALSO already in
the row's actual final context. When that happens and gold_context_fixes_it
is still True (generation works fine on clean gold-only context), the real
cause is distractors alongside the gold doc confusing generation -- that is
mode 6 by the lab's own fix catalogue, not mode 4.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import markdown_chunks  # noqa: E402
from labs.lab3.search import load_corpus, load_questions  # noqa: E402
from labs.lab4.evaluate import build_retriever, judge_correctness  # noqa: E402
from labs.lab4.rag import answer_with_gold_context  # noqa: E402

MODES = {
    1: "missing_content",
    2: "chunk_boundary",
    3: "embedding_mismatch",
    4: "ranking",
    5: "reranker",
    6: "generation",
    7: "presentation",
}

_NUM = re.compile(r"\d+(?:\.\d+)?")


def answer_in_corpus(gold_answer: str, corpus: dict[str, str],
                     relevant_docs: list[str]) -> bool:
    """Mode 1 test -- improved over plain keyword overlap.

    Numbers are the highest-signal, lowest-ambiguity part of most gold
    answers here (durations, ages, amounts): if none of the gold answer's
    numbers appear anywhere in the relevant docs, that's strong independent
    evidence of mode 1, regardless of word overlap.
    """
    text = " ".join(corpus.get(d, "") for d in relevant_docs).lower()
    if not text:
        return False

    gold_nums = set(_NUM.findall(gold_answer))
    text_nums = set(_NUM.findall(text))
    if gold_nums and not (gold_nums & text_nums):
        return False

    tokens = [t for t in gold_answer.lower().split() if len(t) > 4]
    if not tokens:
        return True
    return sum(1 for t in tokens if t.strip(".,;()") in text) / len(tokens) > 0.4


def gold_context_fixes_it(q: dict, corpus: dict[str, str]) -> bool | None:
    """Mode 6 test. Feed the generator ONLY the gold docs and re-ask.

    Direction matters: True means the generator was always capable and
    retrieval starved it (a retrieval failure). False means generation is
    broken even with perfect input.
    """
    gold_texts = [corpus[d] for d in q["relevant_docs"] if d in corpus]
    if not gold_texts:
        return None
    g = answer_with_gold_context(q["question"], gold_texts)
    score = judge_correctness(q["question"], g.text, q["gold_answer"])
    return score >= 2


def in_top_30(retriever, q: dict, k: int = 30) -> bool:
    """Was the gold doc anywhere in a wide retrieval pass?"""
    hits = retriever.search(q["question"], k=k)
    return bool({h.doc_id for h in hits} & set(q["relevant_docs"]))


def in_final(row: dict, q: dict) -> bool:
    """Was the gold doc actually in the final context used to produce THIS
    row's answer? Ground truth from the saved row, not re-derived live.
    """
    return bool(set(q["relevant_docs"]) & set(row.get("retrieved", [])))


def gold_rank(retriever, q: dict, k: int = 30) -> int | None:
    """Rank (1-indexed) of the first relevant doc in a wide search, or None
    if it isn't in the top k at all.
    """
    hits = retriever.search(q["question"], k=k)
    for i, h in enumerate(hits, start=1):
        if h.doc_id in q["relevant_docs"]:
            return i
    return None


def find_gold_chunk_text(corpus: dict[str, str], relevant_docs: list[str],
                         gold_answer: str, size: int = 400) -> str | None:
    """Locate the specific 400-char chunk (same chunking as build_retriever)
    most likely to hold the gold answer, by number + keyword overlap.
    """
    gold_nums = set(_NUM.findall(gold_answer))
    gold_words = {t.strip(".,;()") for t in gold_answer.lower().split() if len(t) > 5}

    best, best_score = None, -1
    for doc_id in relevant_docs:
        text = corpus.get(doc_id, "")
        for chunk in markdown_chunks(text, doc_id, size=size):
            ct = chunk.text.lower()
            score = 3 * sum(1 for n in gold_nums if n in ct)
            score += sum(1 for w in gold_words if w in ct)
            if score > best_score:
                best, best_score = chunk.text, score
    return best


def retrievable_by_own_text(retriever, gold_chunk_text: str | None,
                            relevant_docs: list[str], k: int = 10) -> bool | None:
    """Mode 3 vs 2. Search using the gold chunk's OWN text as the query."""
    if not gold_chunk_text:
        return None
    hits = retriever.search(gold_chunk_text, k=k)
    return bool({h.doc_id for h in hits} & set(relevant_docs))


def classify(row: dict, q: dict, corpus: dict[str, str], *,
             gold_context_fixes_it: bool | None = None,
             in_top_30: bool | None = None,
             already_in_final: bool | None = None,
             retrievable_by_own_text: bool | None = None) -> tuple[int, str]:
    """Walk the T4 §5 diagnostic tree. Returns (mode, evidence)."""

    if row.get("correctness", 0) >= 2 and not row.get("citations_valid", True):
        return 7, f"correct answer, invalid citations {row.get('invalid_citations')}"

    if not answer_in_corpus(q["gold_answer"], corpus, q["relevant_docs"]):
        return 1, "gold answer content not found in the relevant documents"

    if gold_context_fixes_it is False:
        return 6, "PURE: still wrong with gold context in the prompt -- generation failure"

    if in_top_30:
        if already_in_final:
            return 6, ("DISTRACTOR: gold doc WAS in the final context, gold-context-only test "
                      "passed, yet the real answer was wrong -- other chunks confused generation.")
        return 4, (f"gold doc in top 30 but not in the {len(row.get('retrieved', []))} "
                   "chunks the generator saw -- ranking (no reranker stage in this pipeline)")

    if retrievable_by_own_text:
        return 3, "not in top 30, but its own text retrieves it -- query/embedding mismatch"

    return 2, "needs_human_check: not retrievable even by its own text -- open the chunks around the gold answer"

def pareto(tally: Counter) -> str:
    total = sum(tally.values()) or 1
    lines, cum = ["failure mode          n    share   cumulative"], 0
    for mode, n in tally.most_common():
        cum += n
        bar = "█" * round(30 * n / total)
        lines.append(f"{MODES[mode]:<20} {n:>3}   {n/total:>5.1%}   "
                     f"{cum/total:>5.1%}  {bar}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/lab4.json")
    ap.add_argument("--pareto", action="store_true")
    ap.add_argument("--save", default="reports/lab5_diagnosis.json")
    args = ap.parse_args()

    rows = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in load_questions(include_unanswerable=True)}
    corpus = load_corpus()
    retriever = build_retriever()

    failures = [r for r in rows
                if r.get("correctness", 2) < 2 or not r.get("citations_valid", True)]
    print(f"{len(failures)} failures out of {len(rows)}\n")

    out, tally = [], Counter()
    for r in failures:
        q = questions[r["id"]]

        gc_fix = top30 = already_in = by_own_text = None
        is_mode_7 = r.get("correctness", 0) >= 2 and not r.get("citations_valid", True)

        if not is_mode_7 and answer_in_corpus(q["gold_answer"], corpus, q["relevant_docs"]):
            gc_fix = gold_context_fixes_it(q, corpus)
            if gc_fix:
                top30 = in_top_30(retriever, q)
                if top30:
                    already_in = in_final(r, q)
                else:
                    chunk_text = find_gold_chunk_text(corpus, q["relevant_docs"], q["gold_answer"])
                    by_own_text = retrievable_by_own_text(retriever, chunk_text, q["relevant_docs"])

        mode, evidence = classify(
            r, q, corpus,
            gold_context_fixes_it=gc_fix,
            in_top_30=top30,
            already_in_final=already_in,
            retrievable_by_own_text=by_own_text,
        )

        if top30:
            rank = gold_rank(retriever, q)
            evidence += f" (gold doc rank in top 30: {rank}, already in final context: {already_in})"

        tally[mode] += 1
        out.append({"id": r["id"], "kind": q["kind"], "mode": mode,
                    "mode_name": MODES[mode], "evidence": evidence,
                    "question": q["question"], "answer": r["answer"][:300]})
        print(f"  {r['id']:<5} {MODES[mode]:<20} {evidence}")

    print("\n" + pareto(tally))
    print("\nCases marked needs_human_check are Part A2. Open them.")

    p = ROOT / args.save
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()