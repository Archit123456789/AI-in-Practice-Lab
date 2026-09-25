# Lab 4 --- RAG v1: Grounded Answers with Citations

## 1. Objective

The goal of Lab 4 was to build and evaluate a retrieval-augmented
generation (RAG) system that answers using only supplied sources, cites
evidence by source index, refuses when evidence is insufficient, handles
partial support, and meets quality, latency, and cost targets.

The evaluation used a 45-question golden set: 40 answerable and 5
unanswerable questions.

## 2. System Design

The final pipeline is:

`Question → Retrieval → Context formatting → LLM generation → Validation → Repair → Final answer`

The retriever uses the Lab 3 configuration: - Markdown chunking - chunk
size 400 - heading-path information supplied by the Markdown chunker -
dense retrieval

The generation prompt enforces source-only answering, indexed citations,
refusal when evidence is insufficient, disagreement handling, and length
discipline.

### Validation and repair

Generated answers are checked for: 1. truncation; 2. empty answers; 3.
refusal behavior; 4. invalid/out-of-range citations; 5. missing
citations on non-refusal answers.

The repair ladder retries truncated outputs, strips invalid citations,
requests a corrective rewrite for missing citations, and finally falls
back to the exact refusal if the answer remains invalid. Thus malformed
non-refusal answers are not returned.

## 3. Full Evaluation Results

  Metric                                  Result        Target Status
  -------------------------- ------------------- ------------- -----------------------
  Citation validity                        1.000         1.000 PASS
  Faithfulness                             0.978      \>= 0.90 PASS
  Correctness (normalised)                 0.775      \>= 0.75 PASS
  Refusal recall                     1.000 (5/5)      \>= 0.80 PASS
  Refusal precision                  0.833 (5/6)      \>= 0.70 PASS
  p95 latency                            4942 ms   \<= 6000 ms PASS
  Cost/query                   approx. \$0.01094    \<= \$0.01 SLIGHTLY ABOVE TARGET

The full evaluation used 140 model calls, with 77,808 input tokens and
53,727 output tokens. Reported total cost was \$0.4921. p50 latency was
2778 ms and p95 latency was 4942 ms.

### Performance by question type

  Question type     Mean correctness
  --------------- ------------------
  Single-hop                   0.889
  Trap/archived                0.833
  Aggregation                  0.750
  Paraphrase                   0.700
  Multi-hop                    0.600

Multi-hop questions were the weakest category.

## 4. Gold-Context Experiment

To separate retrieval errors from generation errors, the generator was
evaluated with gold/relevant context.

-   Gold-context correctness, A = **0.893**
-   Retrieved-context correctness, B = **0.774**
-   Retrieval-attributable loss = **A - B = 0.119**
-   Generation-attributable loss = **1 - A = 0.107**

Retrieval accounts for slightly more of the observed correctness loss
than generation (0.119 vs. 0.107), but the difference is only 0.012.
Therefore the experiment does not indicate a dominant single failure
source.

The high gold-context score also shows that the generator performs
substantially better when the relevant evidence is supplied directly.

## 5. Human Calibration and Judge Agreement

### Faithfulness rubric

The faithfulness judge evaluates whether claims in the answer are
supported by retrieved context. It does not judge general-world truth or
helpfulness. A partial answer is faithful when the supported portion is
backed by context and the unsupported portion is explicitly declined.

### Correctness rubric

Correctness is scored: - 2 = fully correct/substantively equivalent - 1
= partially correct - 0 = wrong, inappropriate refusal, or answering
when refusal is required

### Human calibration

Twenty questions were manually labelled before using judge scores for
calibration.

The calibration showed that the sampled answers were faithful to their
retrieved contexts, while several were only partially correct because
they omitted details present in the reference answer.

Examples: - Q03 correctly said maternity is unavailable on Bronze but
omitted availability on Silver, Gold, and Platinum. - Q05 correctly gave
the 30-day annual-policy grace period but omitted the 15-day instalment
rule and continuity details. - Q11 correctly gave road ambulance and
Gold air-ambulance amounts but omitted Platinum air-ambulance
coverage. - Q19 correctly calculated the ₹10,000 room limit and 0.667
ratio but omitted the exception that pharmacy and consumables are not
reduced.

### Cohen's kappa

  Judge            Raw agreement   Cohen's kappa    n   Threshold
  -------------- --------------- --------------- ---- -----------
  Faithfulness              1.00           1.000   20     \>= 0.4
  Correctness               0.85           0.625   20     \>= 0.4

Both kappa values exceed the 0.4 threshold. No rubric revision was
required.

## 6. Judge Self-Preference / Limitation

The automated judge uses the same general model family as the generation
system, creating a potential self-preference limitation. Human
calibration partially addresses this concern. The observed kappa values
show alignment with the human labels on the 20-question sample, although
the small sample does not eliminate model-family bias.

## 7. Failure Analysis

The strongest observed failure pattern is incomplete correctness rather
than unsupported factual content. Faithfulness is high at 0.978 while
correctness is lower at 0.775.

The calibration examples reinforce this distinction: an answer can be
fully grounded in retrieved evidence while still being only partially
correct because it omits relevant information from the reference answer.

The weakest categories were multi-hop (0.600) and paraphrase (0.700),
suggesting that combining multiple evidence pieces and preserving all
required details during paraphrasing are important improvement areas.

A complete E3 analysis requires classifying ten individual wrong answers
into the lab's seven specified failure modes. The aggregate results
alone do not contain enough information to assign those ten
classifications reliably without inventing them; those classifications
should therefore be added after inspecting the ten wrong-answer
examples.

## 8. Cost and Latency

-   Total reported cost: **\$0.4921**
-   Approximate cost/query: **\$0.01094**
-   p50 latency: **2778 ms**
-   p95 latency: **4942 ms**

Latency passes the 6000 ms target. Cost/query is slightly above the
\$0.01 target. Future optimisation can focus on reducing unnecessary
generation/repair calls, tightening prompts, and reducing output-token
usage while preserving quality and citation validity.

## 9. Overall Findings

1.  Citation validity is perfect at 1.000.
2.  Faithfulness is strong at 0.978.
3.  Correctness is 0.775, narrowly above the 0.75 target.
4.  Refusal recall is perfect at 5/5 and refusal precision is 0.833.
5.  p95 latency is within target at 4942 ms.
6.  Cost is slightly above target at approximately \$0.01094/query.
7.  Gold-context testing gives retrieval loss 0.119 and generation loss
    0.107; retrieval is only marginally larger.
8.  Multi-hop questions are the weakest category at 0.600.
9.  Human calibration gives kappa 1.000 for faithfulness and 0.625 for
    correctness.
10. Both kappa values exceed 0.4, so rubric revision is not required.

## 10. Conclusion

RAG v1 meets the lab's main quality and latency requirements and
demonstrates strong source-grounded behaviour. Its strongest properties
are citation validity, faithfulness, and refusal recall. The main
remaining weaknesses are a narrow correctness margin, weaker multi-hop
performance, incomplete answers in some cases, and slightly excessive
cost.

The gold-context experiment suggests that both retrieval and generation
contribute to correctness loss, with retrieval accounting for a slightly
larger share. Future work should therefore improve retrieval and
multi-hop evidence combination while reducing unnecessary model calls
and output tokens to bring cost below \$0.01/query.

## 11. Reproducibility Commands

``` bash
python evaluate.py --full --save reports/lab4.json
python evaluate.py --gold-context
python evaluate.py --calibrate
python evaluate.py --kappa
```
