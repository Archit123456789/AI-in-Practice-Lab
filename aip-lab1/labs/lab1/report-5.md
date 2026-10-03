# Lab 1 — The Reliable Extractor · Report

**Setup.** Model `gemini-3.5-flash-lite` (the `SMALL` tier), `temperature=0`, prompt cache on. Prices: $0.30 per million input tokens, $2.50 per million output. All tuning was done on the 60-ticket dev set. The 120-ticket test set was run once, and that run is saved in `reports/lab1_test.json`.

## Part A — how the naive version fails

`v0_naive.py` is the first thing you would write: one prompt, then `json.loads` on the reply. Run over 40 dev tickets, here is how it breaks.

| What went wrong | Tickets (of 40) | Example ticket | Taxonomy (T1 §3) |
|---|---|---|---|
| Reply not valid JSON at all | 0 | — | #5 |
| JSON wrapped in a Markdown code fence | 40 | T0054 (all 40) | #5 malformed output |
| Extra prose before or after the JSON | 0 | — | #5 |
| Valid JSON but a required field missing | 0 | — | #6 |
| `category` value outside the allowed list | 40 | T0054 (all 40) | #6 schema violation |
| `urgency` returned as text instead of a number | 40 | T0054 (all 40) | #6 wrong type |
| `policy_number` invented | 0 | — | #8 hallucination |
| Crash (unhandled exception) | 0 | — | — |

The same three things happen on every ticket: the reply is fenced, the category is off the allowed list, and urgency comes back as a word. Nothing else ever goes wrong — the model never invents a policy number and never drops a field.

The fence alone means `json.loads` parses **0 of 40** replies. Stripping the fence first (a one-line change) gets all 40 to parse, but **still 0 of 40 are fully correct**, because the category and urgency problems were hidden behind the parse error the whole time. You cannot measure output you cannot parse. The run cost $0.0074, p95 latency 1292 ms.

Two of the rows above are not the model misbehaving. The **crash** row is a missing `try` in our own code. The **urgency-as-text** row is not a separate failure — it is the sign that the fence problem and the schema problem are one event: a fenced blob whose contents also break the schema.

## Part B and Part C compared

**B** makes one model call and checks the reply against the full 8-field schema. If anything fails, it returns a record marked for human review instead of raising an error.

**C** is the same, except two fields — `policy_number` and `contains_pii` — are no longer asked of the model. They are found in code with a regex, and the schema sent to the model drops those two fields.

| | v0 | B (dev, 60) | C (dev, 60) | C (test, 120) |
|---|---|---|---|---|
| Valid records | 0 / 40 | 60 / 60 | 60 / 60 | 120 / 120 |
| Field accuracy | — | 0.910 | 0.906 | 0.916 |
| Whole-record accuracy (all 8 right) | 0.00 | 0.467 | 0.450 | 0.525 |
| Cost, whole run | $0.0074 | $0.046 | $0.038 | $0.076 |
| Cost per ticket | $0.00019 | $0.00076 | $0.00063 | $0.00063 |
| p95 latency | 1292 ms | ~1337 ms | 1473 ms | 1389 ms |
| Records flagged for review / crashes | — / 0 | 0 / 0 | 1 / 0 | 0 / 0 |

The B dev cost is computed from the token counts (94,678 in, 6,965 out) because that run was served from cache; its p95 is from an earlier live partial run.

Comparing B and C on the same 60 dev tickets: moving two fields into code cut input tokens by 14%, output tokens by 22%, and cost per ticket by 17%. Field accuracy changed by −0.003 and whole-record accuracy by −0.017, both within the run-to-run noise at 60 tickets. The two fields that moved are now computed by a rule you can read and check, instead of trusted to the model. **Part C's result: cost goes down, accuracy holds.**

Against the assignment targets, the test run passes five of six: valid records 1.00, field accuracy 0.9156 (target 0.90), cost $0.076 (target $0.15), p95 1389 ms (target 4000 ms), zero crashes. It misses whole-record accuracy: 0.525 against a target of 0.55, left as measured. The next two sections are why.

## Per-field accuracy and the category confusion matrix

Accuracy by field on the 120-ticket test run:

| Field | Accuracy | | Field | Accuracy |
|---|---|---|---|---|
| urgency | 0.667 | | category | 0.933 |
| sentiment | 0.808 | | contains_pii | 1.000 |
| escalate | 0.917 | | language, policy_number, product | 1.000 |

Two fields carry all the loss: urgency and sentiment. The rest are 0.917 or better, and four are perfect.

The category confusion matrix (rows are the correct answer, columns are what the model predicted; a dot means zero):

```
              billing claims complaint information policy_change technical
billing          16      .        .          .           .           .
claims            .     21       .          .           .           .
complaint         .      5      11          .           .           .
information       .      3       .         19           .           .
policy_change     .      .       .          .          22           .
technical         .      .       .          .           .          23
```

Every category error — 8 of them — is on one seam: `complaint` or `information` tickets predicted as `claims`. No other pair is ever confused, and the data README already flags this boundary as the ambiguous one. The urgency errors are all off by exactly one step and sit next to a threshold, so the field is mis-calibrated rather than broken.

## Top three error clusters

1. **Urgency thresholds in the wrong place** — about 18 of the ~40 urgency errors on test. The model rates "please send my 80D tax certificate" as 1 when it should be 2 (a document has to be generated), "let me download my e-card" as 2 when it should be 1 (self-service), and "the app crashes when I upload" as 3 when it should be 2 (a bug, not an emergency). **Fix:** add 3–4 worked examples that sit right on each threshold, and start the field description with the test "can this be answered without opening the customer's file?" **Payoff: largest.** Urgency is wrong in roughly 40% of the imperfect records; lifting it from 0.667 to about 0.80 would raise whole-record accuracy from about 0.525 to about 0.60, past the target.

2. **Polite Hinglish and emoji read as urgency and anger** — about 8 urgency errors and 10 sentiment errors. "Jaldi karo" and "kripya" on a routine first request trigger the same-day flag and flip sentiment to frustrated; an emoji does the same (ticket T0060, "...AYUSH? 😡", is marked angry when it should be neutral). **Fix:** one sentence in each field's description plus two examples. **Payoff: medium**, roughly 0.03–0.05 on each field.

3. **The claims / complaint boundary** — all 8 category errors, plus about 6 sentiment errors. "The network hospital refused cashless and said you still owe them" is a complaint about Aurora's handling, but the model sees claim words and says `claims`. "Thanks for settling my claim, just confirming my no-claim bonus" is information, but again the model says `claims`. **Fix:** add one complaint example where claim words appear but the subject is Aurora's conduct, and one information example that follows a settled claim. **Payoff: cheap**, category from 0.933 to about 0.97.

## The economic argument (step 4.5)

- Measured on the test run: $0.076 for 120 tickets, so **$0.00063 per ticket** (about ₹0.053 at ₹83.5 to the dollar).
- At 10,000 tickets a day, that is about **$2,300 a year** to run.
- A person doing the same work takes about 40 seconds a ticket; at ₹300 an hour that is ₹3.33, about **$0.040 a ticket**, or about **$146,000 a year** for the same volume. The model is roughly **60 times cheaper** per ticket.
- **Break-even:** the model is worth running when its cost, plus the cost of people fixing the records it gets wrong, is less than paying people to do everything: `c_llm + (1 − r) · c_human < c_human`. That simplifies to `r > c_llm / c_human`, which is `0.053 / 3.33`, about **1.6%**. We are at 52.5% whole-record accuracy, roughly 30 times above the line. On cost alone this is an easy yes.
- **What actually limits the saving** is not money but review capacity. At 52.5% accuracy, about 4,750 of 10,000 daily records have at least one wrong field. That is only a ~52% cut in agent workload *if you can tell which records those are* — and right now you cannot, because the review flag fired **0 times** on the test set. The real saving stays below 52% until the system can flag its own shaky answers.

## One thing that didn't work

Removing `policy_number` and `contains_pii` from the schema in Part C was supposed to change only the output, not the accuracy. It did not: on dev, sentiment accuracy dropped from 0.900 to 0.833 (4 of 60 tickets changed), even though the sentiment field and its description were untouched. The schema is part of the prompt, so taking two fields out changed the text the model was reading. Overall field accuracy still moved only −0.003 and the cost saving is real, so C stands — but the lesson is that trimming the schema is a prompt change and needs a before-and-after check like any other prompt change.
