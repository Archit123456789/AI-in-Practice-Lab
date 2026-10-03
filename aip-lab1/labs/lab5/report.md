# Lab 5: RAG v2 — Diagnose, Fix, Prove

## Executive Summary

Lab 4 had **6/45 failures**. Failure analysis identified **Mode 6 (generation)** as the dominant cluster (**4/6, 66.7%**).

The selected fix was to activate the unused `final_k` parameter and reduce the generator context from **12 → 5 chunks**. The fix was expected to recover 2/4 Mode-6 failures.

The result was negative: **correctness decreased from 0.925 → 0.875 (-0.050)** and failures increased from **6 → 9**. None of the four targeted failures were recovered. The fix reduced cost and latency but introduced new correctness and refusal-precision regressions.

---

## Part A — Failure Classification

| Mode    | Failure Type       | Questions          |  n |
| ------- | ------------------ | ------------------ | -: |
| 6       | Generation         | Q11, Q23, Q29, Q32 |  4 |
| 4       | Ranking            | Q04                |  1 |
| 3       | Embedding mismatch | Q37                |  1 |
| 1/2/5/7 | —                  | —                  |  0 |

**Mode 6 = 66.7%**, making it the largest failure cluster.

Mode-6 classification used actual `answer_with_gold_context()` regeneration and re-judging. A failure was classified as Mode 6 only when gold context did **not** fix it.

An early bug in `in_top_30` was corrected. The original check tested document-level presence, which could give false positives because relevant documents contain many similar chunks. The final check identifies the **specific fact-bearing gold chunk** using lexical overlap and tracks its rank.

The Mode-1 corpus check was also fixed because the original implementation dropped tokens ≤4 characters, potentially removing important numerical values.

**Human review required: none.**

---

## Part B — Fix Selection

| Cluster             |  n | Fix                  | Expected Recovery | Effort   |
| ------------------- | -: | -------------------- | ----------------: | -------- |
| Mode 6 — Generation |  4 | `final_k`: 12 → 5    |               2/4 | Trivial  |
| Mode 4 — Ranking    |  1 | Retrieval k: 12 → 30 |             0–1/1 | Trivial  |
| Mode 3 — Embedding  |  1 | Hybrid BM25 + dense  |             0–1/1 | Moderate |

**Selected: Mode 6**

Reason: it was the largest failure cluster and the cheapest intervention. `final_k` was previously unused, so activating it required only a minimal change.

### Prediction

Expected `final_k=5` to recover **Q04 and Q23**, while Q11 and the other failure modes would remain unchanged.

---

## Part C — Implementation

Only `labs/lab4/rag.py::answer_question()` was changed:

```python
hits = retriever.search(question, k=k)[:final_k]
```

Retrieval remained at **k=12**; only the context passed to the generator was reduced to the top 5 chunks.

No changes were made to retrieval, chunking, embeddings, prompts, or evaluation.

---

## Part D — Results

### Before vs After

| Metric                      |      v1 |      v2 |           Δ |
| --------------------------- | ------: | ------: | ----------: |
| Correctness                 |   0.925 |   0.875 |  **-0.050** |
| Faithfulness                |   0.933 |   0.956 |      +0.023 |
| Citation validity           |   1.000 |   1.000 |       0.000 |
| Refusal recall (full)       |   0.600 |   0.800 |      +0.200 |
| Refusal precision (full)    |   1.000 |   0.800 |  **-0.200** |
| Refusal precision (partial) |   1.000 |   0.625 |  **-0.375** |
| nDCG@10                     |   0.860 |   0.860 |       0.000 |
| Recall@5                    |   0.903 |   0.903 |       0.000 |
| Cost/query                  | $0.0122 | $0.0108 | **-0.0014** |
| p95 latency                 | 5463 ms | 5248 ms | **-215 ms** |

### Failure Reclassification

| Mode                        |    v1 |    v2 |
| --------------------------- | ----: | ----: |
| Mode 6 — Generation         |     4 | **6** |
| Mode 4 — Ranking            |     1 | **2** |
| Mode 3 — Embedding mismatch |     1 |     1 |
| **Total**                   | **6** | **9** |

None of the four targeted failures were recovered.

New failures:

* **Q20, Q35:** new Mode-6 failures
* **Q44:** new Mode-4 failure because its required evidence fell outside the top-5 context

Q37 remained unchanged, as expected, because the embedding/retrieval mechanism was not modified.

---

## Part E — Regression and Noise

The main regression was correctness:

**0.925 → 0.875 (-0.050)**

Refusal precision also decreased substantially. A separate `--strict` run using the same v2 code produced different refusal metrics (**0.600/1.000** vs canonical **0.800/0.800**), demonstrating LLM sampling variance. Therefore, refusal deltas should be interpreted cautiously at this sample size.

The correctness regression is the primary result.

---

## Part F — Next Fix

Mode 6 remains dominant (**6/9 failures**), so generation remains the main area for investigation.

Instead of removing context, the next experiment should:

> **Keep all 12 retrieved chunks but reorder them so the highest-scoring evidence appears first.**

This tests whether context ordering improves generation without sacrificing retrieval coverage.

---

## Conclusion

The `final_k: 12 → 5` fix was **diagnosed correctly but did not work**.

* Correctness: **-0.050**
* Failures: **6 → 9**
* Targeted failures recovered: **0/4**
* Cost/query: **-11.5%**
* p95 latency: **-215 ms**

The experiment shows that reducing context improved efficiency but **harmed answer correctness**. The next step should preserve retrieval coverage and investigate context ordering or generation quality directly.
