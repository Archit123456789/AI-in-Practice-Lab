# AIP Lab 2 — Evaluation Report

## Part A — Few-shot selection

I selected six edge-case demonstrations covering: billing/complaint boundaries, requests without a policy number, Hinglish input, satisfied-but-urgent cases, policy numbers appearing only in quoted replies, and one difficult case from Lab 1.

The selected examples were useful because they target cases where the classifier must distinguish semantic intent, urgency, sentiment, and escalation rather than relying only on keywords.

**A4 — selection bias/leakage:** the examples were selected from the dev set and evaluated on that same dev set. This creates selection bias and means the few-shot improvement cannot be treated as an unbiased generalization result. A better protocol is to select demonstrations from training data (or a separate selection split) and reserve dev/test data strictly for evaluation.

## Part B — Grid results

| Config | Record acc. | Field acc. | Schema valid | Cost | p95 |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 45.0% | 89.38% | 100% | $0* | 0 ms* |
| Zero-shot MAIN | 5.0% | 47.29% | 100% | $0.0039 | 5291 ms |
| Few-shot | **50.0%** | **92.08%** | 100% | $0* | 0 ms* |
| Few-shot MAIN | 0.0% | 42.29% | 100% | $0* | 0 ms* |
| Few-shot + reasoning | 45.0% | 89.58% | 100% | $0.0120 | 30,824 ms |
| Few-shot + reasoning MAIN | 0.0% | 42.29% | 100% | $0* | 0 ms* |
| Cascade | 41.67% | 85.83% | 100% | $0.0117 | 2571 ms |

*The zero-cost/zero-latency values are cache artifacts from this run, not production economics.

**Q1 — Which knob moved numbers more?**  
Prompting moved the observed result more favorably than adding the MAIN model tier: few-shot improved record accuracy from 45.0% to 50.0% and field accuracy from 89.38% to 92.08%. The MAIN variants were substantially worse in this run, but those results were contaminated by Gemini free-tier rate limiting (429 errors), so they should not be interpreted as a clean model-tier comparison.

**Q2 — What did reasoning cost and what did it buy?**  
On the few-shot configuration, reasoning changed record accuracy from 50.0% to 45.0% and field accuracy from 92.08% to 89.58%, while measured cost increased to $0.0120 and p95 latency increased to about 30.8 s. Thus, in this experiment, reasoning bought no measured quality improvement; it was a negative result.

**Q3 — Is any configuration dominated?**  
The grid's automatic dominance output flags several configurations, but because the MAIN results were affected by rate limiting and several zero-cost/zero-latency values were cache artifacts, I would not use that automatic dominance list as a production conclusion. The clearest practical conclusion is that reasoning was strictly worse than few-shot on the measured quality, cost, and latency axes.

## Part C — Cascade

The cascade escalated **6.67%** of the 60 dev cases. Its measured blended result was **41.67% record accuracy**, **85.83% field accuracy**, and **$0.0117 total cost**, or approximately **$0.19 per 1,000 tickets**. At 10,000 tickets/day, that corresponds to approximately **$710/year** at the measured cost rate.

The cascade did not provide a statistically significant improvement over zero-shot: the paired comparison gave **b = 2, c = 0, p = 0.5000**.

The two-sample disagreement trigger also has limited value when both samples use temperature 0, because cached/deterministic outputs can be identical. A useful disagreement test requires a genuinely different sample (for example, a nonzero temperature on the second sample).

## Part D — Paired test

For zero-shot vs few-shot:

- **b = 5:** zero-shot correct, few-shot wrong
- **c = 8:** few-shot correct, zero-shot wrong
- **p = 0.5811**

Therefore, the observed 5-point record-accuracy gain from few-shot is **not statistically significant** on this 60-case paired evaluation.

For zero-shot vs reasoning:

- **b = 10**
- **c = 10**
- **p = 1.0000**

This supports the conclusion that reasoning did not produce a measurable improvement.

## Part E — Error analysis

I reviewed 20 failures from the few-shot run. The dominant failure pattern was **urgency classification**, with **28 urgency errors overall** in the 60-case evaluation. The next recurring field-level issues were **sentiment (6 errors)** and **escalation (3 errors)**; these counts are field errors and can overlap within the same record.

The urgency confusion matrix below uses rows = gold urgency and columns = predicted urgency:

| Gold \ Pred | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| **1** | 9 | 3 | 0 | 0 | 0 |
| **2** | 6 | 8 | 2 | 0 | 0 |
| **3** | 1 | 6 | 4 | 0 | 0 |
| **4** | 0 | 1 | 2 | 8 | 3 |
| **5** | 0 | 0 | 0 | 4 | 3 |

This shows a strong tendency to compress urgency toward the middle: urgency 2 is often predicted as 1, urgency 3 as 2, and urgency 5 as 4. High-urgency cases can therefore be under-called, which is operationally more important than a small aggregate accuracy difference.

### Recommendation

**Recommend the few-shot configuration as the best starting point for this lab's measured setup**, because it had the highest observed record accuracy (50.0%) and field accuracy (92.08%) among the cleanly interpretable non-MAIN configurations, while avoiding the large latency/cost penalty of reasoning.

However, this is a **provisional recommendation**, not evidence of production superiority: its apparent $0 cost and 0 ms latency are cache artifacts, and the few-shot examples were selected from the evaluation dev set.

**What would change my mind:** I would switch away from few-shot if a leakage-free holdout/test evaluation showed that another configuration produced a materially higher record/field accuracy with acceptable cost and latency, or if a calibrated production evaluation showed that few-shot's urgency under-calling created unacceptable operational risk.

## Negative result

The clearest negative result is that **adding reasoning did not help**: few-shot reasoning fell from 50.0% to 45.0% record accuracy and from 92.08% to 89.58% field accuracy, while measured cost rose to $0.0120 and p95 latency rose to approximately 30.8 seconds. This is evidence against adding reasoning by default for this task.
