Lab 1 — The Reliable Extractor

1. Part A — v0 failure table

The naive extractor was run on 40 tickets. The main failure was that the chat model returned JSON inside Markdown fences, while v0 passed the entire response directly to json.loads. Rate limiting also caused unhandled exceptions.

Failure mode

Count in 40

Example

Markdown fence around JSON

29

T0054

Rate-limit exception

11

T0078

Urgency returned as a string (after fence removal)

29

T0054

Category outside allowed set (after fence removal)

29

T0054

Thus v0 produced 0/40 clean records. Removing the Markdown fence exposed further schema/type problems; it did not make the records reliable.

2. Variant comparison

Variant

Quality / field accuracy

Cost (120 test)

p95 latency

v0

0/40 clean

N/A

N/A

B

0.8987

$0.0157

1463 ms

C

0.9083

$0.0133

3314 ms

C improves field accuracy over B by about 0.96 percentage points and reduces measured cost by about 15%, at the expense of higher p95 latency. Both B and C achieved 100% schema validity.

3. Per-field accuracy — C test

Field

Accuracy

urgency

0.592

sentiment

0.825

category

0.917

escalate

0.933

contains_pii

1.000

language

1.000

policy_number

1.000

product

1.000

The main weakness is urgency. The deterministic fields reach 100%, which is the main engineering benefit of Part C.

Category confusion matrix — C test

Rows are gold labels; columns are predictions.

Gold \ Pred.

billing

claims

complaint

information

policy_change

technical

billing

16

0

0

0

0

0

claims

0

20

0

1

0

0

complaint

2

5

9

0

0

0

information

0

0

0

22

0

0

policy_change

0

0

0

0

22

0

technical

0

0

0

2

0

21

The largest category problem is the complaint boundary: complaint tickets are sometimes classified as claims or billing. Technical tickets are also occasionally classified as information.

4. Top three error clusters and fixes

1. Urgency boundary errors

Urgency is the weakest field at 0.592. The failures are concentrated around distinctions such as 1/2, 3/4 and 4/5 rather than being random.

Fix: make the urgency prompt more decision-tree-like, explicitly checking: customer lookup/action required → 2; something already stuck → 3; repeated failure or explicit threat → 4; emergency/formal Ombudsman escalation → 5. Add explicit tense examples for “going to the Ombudsman” versus “I am filing”.

2. Complaint vs. claim/billing

The confusion matrix shows complaint → claims (5) and complaint → billing (2). The issue is whether the subject is Aurora's conduct or the underlying transaction.

Fix: add paired counterexamples: “my claim was rejected” → claims, versus “your agent mishandled/ignored/mis-sold this” → complaint.

3. Technical vs. information

Two technical tickets were classified as information.

Fix: explicitly prioritize a broken app/portal/OTP/upload over the fact that the customer is asking a question. “What do I do?” about a broken system remains technical.

5. D5 — economic argument

Measured C cost on the 120-ticket test run:

Cost per ticket = $0.0133 / 120 = $0.0001108

At 10,000 tickets/day: about $1.11/day

Annual model cost = about $404/year

Human processing:

40 seconds/ticket × 10,000 tickets/day = 111.1 agent-hours/day

At ₹300/hour = ₹33,333/day

Annual human cost = ₹12.17 million/year (about ₹1.22 crore)

Even allowing for human correction of incorrect automated records, the model cost is tiny relative to the manual baseline. Under the simple assumption that every incorrect record requires the full 40-second manual effort, the break-even record accuracy is approximately:

automation_cost / human_cost ≈ 404 / (12.17 million / 83.5 INR per USD) ≈ 0.28%

So the economic case is overwhelmingly positive at the measured accuracy. The practical deployment decision should therefore be driven much more by error severity, reliability, privacy and review workflow than raw inference cost.

6. What did not work

A first attempt was to let the LLM decide all fields in Part B, including policy number, PII and escalation. It was valid and reasonably accurate, but Part C showed why this is the wrong boundary: deterministic fields are better handled by code.

The clearest negative result was that moving deterministic fields out did not improve judgement accuracy dramatically. C's field accuracy was only modestly higher than B (0.9083 vs 0.8987), while p95 latency increased. This is not a failure of the architecture: the purpose of Part C is primarily to make deterministic fields auditable and 100% reliable and to reduce cost, not to make the LLM better at subjective judgement.

Conclusion

Part C is the better production design: LLM for judgement, Python for deterministic rules, Pydantic for validation. It achieved 100% on policy number, PII, product and language, maintained 100% schema validity, improved overall field accuracy over B, and reduced cost. The next improvement should target the urgency and complaint-boundary errors rather than adding more fields to the model.