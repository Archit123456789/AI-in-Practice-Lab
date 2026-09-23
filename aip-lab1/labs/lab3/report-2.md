# Lab 3 Report — Semantic Search for Aurora

n = 42 (Q36, Q38, Q39 excluded — no relevant document, so recall is undefined for them).

---

## A. Chunking

### A1 — strategy @ size=800

| config | hit_rate@1 | hit_rate@5 | recall@5 | MRR | nDCG@10 | chunks | build (s) |
|---|---|---|---|---|---|---|---|
| fixed | 0.7381 | 0.9524 | 0.8373 | 0.8387 | 0.7952 | 83 | 0.03 |
| sliding | 0.7857 | 0.9286 | 0.8452 | 0.8451 | 0.8053 | 91 | 0.02 |
| recursive | 0.7619 | 0.9524 | 0.8750 | 0.8611 | 0.8251 | 98 | 0.03 |
| **markdown** | 0.7619 | 0.9762 | 0.8988 | 0.8720 | **0.8458** | 164 | 0.04 |

Markdown-aware chunking wins on nDCG@10 and recall@5, at ~2x the chunk count of `fixed` and negligible extra build cost. Winner carried forward: **markdown**.

### A2 — size sweep on markdown

| config | hit_rate@1 | recall@5 | MRR | nDCG@10 | chunks |
|---|---|---|---|---|---|
| **markdown-400** | 0.7857 | 0.9028 | 0.8800 | **0.8527** | 235 |
| markdown-800 | 0.7619 | 0.8988 | 0.8720 | 0.8458 | 164 |
| markdown-1600 | 0.7143 | 0.8750 | 0.8262 | 0.8075 | 150 |

Not monotonic, and smaller wins: 400 > 800 > 1600 on every metric. This is the dilution effect — a larger chunk averages in more off-topic sentences, pulling its embedding away from any single question's intent, so precision degrades even as each chunk covers more raw text. The cost is more chunks to index (235 vs 150), which is irrelevant at this corpus size and only matters at the D2 scale. Winner: **markdown, 400 chars**.

### A3 — heading-path prefix ablation (on markdown-800)

| config | hit_rate@1 | hit_rate@5 | recall@5 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| with prefix | 0.7619 | 0.9762 | 0.8988 | 0.8720 | **0.8458** |
| without prefix | 0.6190 | 1.0000 | 0.9048 | 0.7837 | 0.7915 |

**nDCG@10 delta: +0.0543** — the prefix helps, but not uniformly. It raises hit_rate@1 (0.62→0.76) and nDCG@10 sharply, while *lowering* hit_rate@5 and recall@5 slightly. The prefix sharpens ranking — it pulls the correct chunk closer to the top — at a small cost to raw recall, since it can nudge a marginally relevant chunk out of the top-5 window in favor of a better-ranked one. Net effect is positive because ranking quality (nDCG, hit@1, MRR) is what the deployment cares about.

### A4 — a chunking failure

*[Fill in: pick one golden question your winning config (markdown-400, dense) gets wrong. Run `retriever.search(question, k=5)` and print the returned chunks next to the chunk that should have matched. One paragraph on why the chunk boundary caused the miss.]*

---

## B. Dense vs BM25 vs Hybrid

All on markdown-400.

### B1 — overall

| config | hit_rate@1 | hit_rate@5 | recall@5 | MRR | nDCG@10 | p95 ms |
|---|---|---|---|---|---|---|
| **dense** | 0.7857 | 0.9762 | 0.9028 | 0.8800 | **0.8527** | 0.40 |
| bm25 | 0.4762 | 0.9286 | 0.7956 | 0.6698 | 0.6978 | 0.28 |
| hybrid | 0.6667 | 0.9762 | 0.8631 | 0.7976 | 0.7949 | 0.58 |

### B2 — per-kind MRR (not hit_rate@5, which is saturated 0.93–0.98 across all three)

| kind (n) | dense | bm25 | hybrid |
|---|---|---|---|
| aggregation (4) | 0.8750 | 0.3750 | 0.5833 |
| multi_hop (10) | 1.0000 | 0.6500 | 0.8167 |
| paraphrase (5) | 0.8000 | 0.4867 | 0.6500 |
| single_hop (18) | 0.9074 | 0.8519 | 0.9444 |
| trap_archived (3) | 0.8333 | 0.5111 | 0.6667 |
| unanswerable (2) | 0.3125 | 0.4167 | 0.3750 |

Dense wins on nearly every kind, BM25 is only competitive on `single_hop`. **Paraphrase MRR (dense) = 0.80**, above the 0.75 target.

### Q44 / Q41 mechanism

| | dense | bm25 | hybrid |
|---|---|---|---|
| Q44 (`AUR-HI-SIL-2026`, exact identifier) | 0.50 | **1.00** | 1.00 |
| Q41 (paraphrased grace-period question, no lexical overlap) | **1.00** | 0.00 | 0.25 |

Q44 is a pure lexical match — an exact code string has no reason to sit near anything semantically, so BM25 wins outright. Q41 has zero token overlap with the source text, so only the dense embedding can bridge it. Fusion rescues Q44 to 1.00 but *drags Q41 down to 0.25* — it damaged the question dense had solved outright.

### B3 — RRF k sweep

| k | 10 | 30 | 60 | 100 |
|---|---|---|---|---|
| nDCG@10 | 0.8156 | 0.7949 | 0.7949 | 0.7901 |

Small effect, as expected — RRF is deliberately insensitive to k, which is why it's a safe default.

### B4 — fusion weights

| weights (dense:bm25) | 1:1 | 2:1 | 1:2 | 3:1 |
|---|---|---|---|---|
| nDCG@10 | 0.7949 | 0.8068 | 0.7926 | 0.8136 |

Even 3:1 weighting (0.8136) doesn't recover plain dense (0.8527) — no fusion configuration beats dense alone on this corpus.

### B5 — hybrid is worse

Dense alone (nDCG@10 0.8527) beats hybrid (0.7949) by a wide margin, at every RRF k and every weighting tried. The per-kind table shows why: dense wins on 5 of 6 kinds, so fusing in a substantially weaker BM25 signal drags more good rankings down than it rescues. This contradicts the general claim that hybrid retrieval is close to a free win — it is corpus-dependent, and on this corpus with a strong embedding model, it loses. **Recommended retriever: dense alone.**

---

## C. Reranking

*[Pending — LLM reranker sweep in progress at time of writing. Fill in once `python search.py --sweep rerank` completes:]*

| Config | nDCG@5 | hit_rate@1 | p95 ms | $/1k queries |
|---|---|---|---|---|
| none (baseline) | | | | $0 |
| cross-encoder | | | | $0 (local model) |
| LLM reranker | | | | |

**Deployment answers (fill in once table is complete):**
- (a) Interactive agent-facing search box: _______
- (b) Overnight batch job: _______
- They should differ because interactive traffic is latency-constrained per query, while batch jobs can trade latency for quality/cost with no user waiting.

**C4 — a regression:** *[Find one question where cross-encoder rerank lowered MRR vs the unranked baseline; diagnose why.]*

---

## D. Index and Metadata

### D1 — exact vs Chroma HNSW (at ~235 chunks)

| config | hit_rate@1 | recall@5 | nDCG@10 | p95 ms |
|---|---|---|---|---|
| exact NumPy | 0.7857 | 0.9028 | 0.8527 | 0.45 |
| Chroma HNSW | 0.7857 | 0.9028 | 0.8527 | 1.38 |

Quality gap: **0.0000** on both recall@5 and nDCG@10 — at this scale HNSW's approximate search finds exactly what exact search finds. The only difference is latency: **HNSW is ~3x slower than exact search here**, because its graph-traversal overhead exceeds a single brute-force matmul at this corpus size.

### D2 — scaling

| Scale | Chunks | Exact (ms) | HNSW (ms) | HNSW/Exact |
|---|---|---|---|---|
| base | 235 | 0.236 | 1.353 | 5.73x |
| ~4k | *[pending — run `python scripts/expand_corpus.py --docs 4000`]* | | | |
| ~40k | *[pending]* | | | |

*[Once the two scaled rows are filled: identify the crossover point where HNSW/Exact drops below 1x, and explain that exact search's cost grows linearly with corpus size (O(n) per query) while HNSW's grows sub-linearly (O(log n)) — the crossover is where HNSW's index-traversal overhead is finally paid back by that better scaling.]*

### D3 — metadata filtering (Q29–Q31)

| Question | hit@1 before | hit@1 after | delta |
|---|---|---|---|
| Q29 | 1.0000 | 1.0000 | 0.0000 |
| **Q30** | **0.0000** | **1.0000** | **+1.0000** |
| Q31 | 1.0000 | 1.0000 | 0.0000 |

Q30's top-ranked hit was an archived document pulling the correct current one out of rank 1. Filtering `status: archived` at query time removes it from candidacy and lets the correct chunk surface — a full recovery with **zero change to the retriever itself**. This is the lab's key implication: when retrieval quality is poor, check data hygiene (stale/duplicate/conflicting documents in the index) before reaching for a better model or a fancier retrieval algorithm.

---

## Final recommended configuration

- **Chunking:** markdown-aware, 400 characters, heading-path prefix retained
- **Retrieval:** dense only (no hybrid — see B5)
- **Reranking:** *[pending C]*
- **Index:** exact NumPy for this corpus size; switch to Chroma HNSW only once corpus scale crosses the D2 crossover point
- **Metadata:** filter `status=archived` at query time

**Numbers at this configuration (dense, markdown-400, no rerank):** nDCG@10 = 0.8527, recall@5 = 0.9028, hit_rate@1 = 0.7857, paraphrase MRR = 0.80 — already clears all six lab targets (nDCG@10 ≥0.80, recall@5 ≥0.85, hit_rate@1 ≥0.65, paraphrase MRR ≥0.75) without hybrid or reranking. *[Update if reranking changes the final pick.]*

## One thing that surprised me

*[Pick one: hybrid retrieval underperforming dense alone despite being the standard advice.]
