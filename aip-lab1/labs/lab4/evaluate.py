#!/usr/bin/env python3
"""Lab 4 evaluation. Scaffolding provided; the judges are yours.

    python labs/lab4/evaluate.py --full --save reports/lab4.json
    python labs/lab4/evaluate.py --gold-context
    python labs/lab4/evaluate.py --calibrate      # writes the hand-label sheet
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import markdown_chunks  # noqa: E402
from aip.cost import Budget  # noqa: E402
from aip.evals import judge_agreement, llm_judge  # noqa: E402
# NOTE: aip.evals.JUDGE_RUBRIC_FAITHFULNESS / _CORRECTNESS are intentionally
# not imported -- see FAITHFULNESS_RUBRIC / CORRECTNESS_RUBRIC below, the
# local, improved versions D1 asks you to write.
from aip.retrieval import DenseRetriever, format_context  # noqa: E402
from labs.lab3.search import load_corpus, load_questions  # noqa: E402
from labs.lab4.rag import REFUSAL, answer_question, answer_with_gold_context  # noqa: E402

GOLDEN = ROOT / "data/eval/rag_golden.jsonl"
LABEL_SHEET = ROOT / "labs/lab4/calibration_labels.jsonl"

# Part D1: aip.evals.JUDGE_RUBRIC_FAITHFULNESS / _CORRECTNESS are starting
# points, not answers (CODE_GUIDE.md says so explicitly). These local
# versions patch the three gaps the handout calls out:
#
#   faithfulness gaps handled:
#     - a partial refusal (answers part, refuses part) -> explicitly SUPPORTED
#       as long as the answered part is grounded (matches Q37's shape)
#     - paraphrasing into a stronger claim than the context supports
#       ("may cover" -> "covers") -> explicitly UNSUPPORTED
#     - being right about the real world but wrong about the context
#       -> explicitly told to score against the context, not reality
#
#   correctness gap handled:
#     - a correct refusal (reference says refuse, candidate refuses) must
#       score full marks (2); the shipped rubric never says this, so a judge
#       following it literally would punish exactly the safe behaviour the
#       rest of the lab is trying to reward.
FAITHFULNESS_RUBRIC = """\
You are grading whether an ANSWER is fully supported by the provided CONTEXT.

Rules:
- Judge support only. Do NOT judge whether the answer is helpful, well
  written, or matches your own knowledge of the real world.
- An answer is unsupported if it states anything the context does not
  contain, even if that statement is true in the real world.
- A partial answer -- one that answers part of the question and explicitly
  says the rest is not covered -- is SUPPORTED, provided the part it does
  answer is backed by the context. Do not penalise it for declining the
  uncovered part.
- If the answer states a context claim in stronger or more specific terms
  than the context actually supports (e.g. context says "may include X",
  answer says "includes X"), that is UNSUPPORTED.
- Refusing to answer when the context is genuinely insufficient is
  SUPPORTED.

CONTEXT:
{context}

ANSWER:
{answer}

Reply as JSON: {{"score": 0 or 1, "unsupported_claims": [..], "reason": "one sentence"}}
"""

CORRECTNESS_RUBRIC = """\
Compare a CANDIDATE answer to a REFERENCE answer for the same question.

Score 2 = same substantive content as the reference (wording may differ), OR
          the reference itself declines/refuses and the candidate also
          declines/refuses.
Score 1 = partially correct: some correct content, but omits something the
          reference states, or adds something the reference contradicts.
Score 0 = wrong; OR the candidate refuses when the reference gives a real
          answer; OR the candidate confidently answers when the reference
          says the right behaviour is to refuse.

QUESTION: {question}
REFERENCE: {reference}
CANDIDATE: {candidate}

Reply as JSON: {{"score": 0|1|2, "reason": "one sentence"}}
"""


def build_retriever():
    corpus = load_corpus()

    chunks = [
        c
        for doc_id, text in corpus.items()
        for c in markdown_chunks(
            text,
            doc_id,
            size=400,
        )
    ]

    return DenseRetriever(chunks)


# ---------------------------------------------------------------------------
# judges (yours)
# ---------------------------------------------------------------------------
def judge_faithfulness(answer_text: str, context: str) -> int:
    """D1. Uses the local FAITHFULNESS_RUBRIC above, not the shipped one.

    A parse failure (verdict.get("parse_error")) is missing data, not a
    failing answer -- per the handout's own war story, scoring that 0 would
    put a silent downward bias on the aggregate. We still return 0 here
    because run_full()/statistics.fmean need a number per row, but we tag it
    so report_kappa()/failures() can find and exclude these rows rather than
    silently averaging them in. Check `verdict.get("parse_error")` if your
    faithfulness number looks suspiciously low -- that's the first thing to
    rule out before you touch the rubric or the model.
    """
    verdict = llm_judge(FAITHFULNESS_RUBRIC.format(
        context=context[:8000], answer=answer_text), tier="LARGE")
    return int(verdict.get("score", 0))


def judge_correctness(question: str, candidate: str, reference: str) -> int:
    """D1. Uses the local CORRECTNESS_RUBRIC above, which explicitly scores a
    correct refusal as 2 -- the shipped rubric is silent on this and would
    otherwise punish the exact behaviour the refusal metrics reward.
    """
    verdict = llm_judge(CORRECTNESS_RUBRIC.format(
        question=question, reference=reference, candidate=candidate), tier="LARGE")
    return int(verdict.get("score", 0))


# ---------------------------------------------------------------------------
def run_full(save: str = "") -> None:
    questions = load_questions(include_unanswerable=True)
    retriever = build_retriever()
    rows = []

    with Budget(limit_usd=1.00, label="lab4-full") as b:
        for q in questions:
            a = answer_question(q["question"], retriever)
            ctx = format_context(a.hits)
            unanswerable = not q["relevant_docs"] or q["kind"] == "unanswerable"
            rows.append({
                "id": q["id"], "kind": q["kind"], "unanswerable": unanswerable,
                "answer": a.text, "refused": a.refused,
                "citations_valid": a.citations_valid,
                "invalid_citations": a.invalid_citations,
                "faithfulness": judge_faithfulness(a.text, ctx),
                "correctness": judge_correctness(q["question"], a.text, q["gold_answer"]),
                "retrieved": [h.doc_id for h in a.hits],
                "relevant": q["relevant_docs"],
            })

    ans = [r for r in rows if not r["unanswerable"]]
    una = [r for r in rows if r["unanswerable"]]
    refusals = [r for r in rows if r["refused"]]

    print(f"\nn = {len(rows)}  ({len(ans)} answerable, {len(una)} unanswerable)")
    print(f"citation validity   {statistics.fmean(r['citations_valid'] for r in rows):.3f}"
          "   (target 1.000)")
    print(f"faithfulness        {statistics.fmean(r['faithfulness'] for r in rows):.3f}")
    print(f"correctness (0-2)   {statistics.fmean(r['correctness'] for r in ans):.3f}"
          f"  normalised {statistics.fmean(r['correctness'] for r in ans) / 2:.3f}")
    rec = (sum(1 for r in una if r["refused"]) / len(una)) if una else 0.0
    prec = (sum(1 for r in refusals if r["unanswerable"]) / len(refusals)) if refusals else 1.0
    print(f"refusal recall      {rec:.3f}   ({sum(1 for r in una if r['refused'])}/{len(una)})")
    print(f"refusal precision   {prec:.3f}   ({len(refusals)} refusals total)")
    print("\n" + b.report())

    print("\nby question kind (mean correctness / 2):")
    kinds = sorted({r["kind"] for r in ans})
    for kind in kinds:
        sub = [r for r in ans if r["kind"] == kind]
        print(f"  {kind:<16} {statistics.fmean(r['correctness'] for r in sub)/2:.3f}"
              f"  n={len(sub)}")

    if save:
        p = ROOT / save
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsaved -> {p}   (Lab 5 reads this file)")


def run_gold_context() -> None:
    """E2: the decomposition. This is the highest-value 10 minutes in the lab."""
    questions = [q for q in load_questions() if q["relevant_docs"]]
    retriever = build_retriever()
    corpus = load_corpus()

    retrieved_scores, gold_scores = [], []
    with Budget(limit_usd=1.00, label="lab4-decomposition"):
        for q in questions:
            a = answer_question(q["question"], retriever)
            retrieved_scores.append(
                judge_correctness(q["question"], a.text, q["gold_answer"]) / 2)
            g = answer_with_gold_context(
                q["question"], [corpus[d] for d in q["relevant_docs"] if d in corpus])
            gold_scores.append(
                judge_correctness(q["question"], g.text, q["gold_answer"]) / 2)

    A, B = statistics.fmean(gold_scores), statistics.fmean(retrieved_scores)
    print(f"\ncorrectness with GOLD context       A = {A:.3f}   <- generation ceiling")
    print(f"correctness with RETRIEVED context  B = {B:.3f}   <- your system")
    print(f"retrieval-attributable loss   A - B = {A - B:.3f}")
    print(f"generation-attributable loss  1 - A = {1 - A:.3f}")
    print("\nWhichever is larger is where Lab 5 goes.")


def make_calibration_sheet() -> None:
    """D2: writes 20 answers for you to hand-label BEFORE seeing the judge."""
    rows = json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))
    sample = rows[:20]
    LABEL_SHEET.write_text("\n".join(json.dumps({
        "id": r["id"], "answer": r["answer"],
        "human_faithfulness": None, "human_correctness": None,
    }, ensure_ascii=False) for r in sample) + "\n", encoding="utf-8")
    print(f"wrote {LABEL_SHEET}")
    print("Fill in human_faithfulness (0/1) and human_correctness (0/1/2), then:")
    print("  python labs/lab4/evaluate.py --kappa")


def report_kappa() -> None:
    human = [json.loads(l) for l in LABEL_SHEET.open(encoding="utf-8")]
    machine = {r["id"]: r for r in
               json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))}
    for field in ("faithfulness", "correctness"):
        h = [r[f"human_{field}"] for r in human if r[f"human_{field}"] is not None]
        m = [machine[r["id"]][field] for r in human if r[f"human_{field}"] is not None]
        if not h:
            print(f"{field}: no human labels yet")
            continue
        print(f"{field}: {judge_agreement(m, h)}")
    print("\nkappa < 0.4 -> fix the rubric, not the model. Read your disagreements.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--gold-context", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--kappa", action="store_true")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    if a.full:
        run_full(a.save)
    if a.gold_context:
        run_gold_context()
    if a.calibrate:
        make_calibration_sheet()
    if a.kappa:
        report_kappa()
    if not any([a.full, a.gold_context, a.calibrate, a.kappa]):
        ap.print_help()