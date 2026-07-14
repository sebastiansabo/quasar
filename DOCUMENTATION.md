# Quasar — Technical Documentation

**Version:** 0.1.0
**Package:** `quasar-context`
**Repository:** https://github.com/sebastiansabo/quasar
**License:** MIT

---

## Table of contents

1. [What Quasar does](#1-what-quasar-does)
2. [Installation](#2-installation)
3. [Quick start](#3-quick-start)
4. [Architecture](#4-architecture)
5. [The faithfulness contract](#5-the-faithfulness-contract)
6. [API reference](#6-api-reference)
7. [Critical value detection](#7-critical-value-detection)
8. [Integration patterns](#8-integration-patterns)
9. [Performance](#9-performance)
10. [Benchmarks](#10-benchmarks)
11. [Limitations](#11-limitations)
12. [Monitoring in production](#12-monitoring-in-production)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. What Quasar does

Quasar is a **context compression layer** that sits between retrieval and the LLM
call in a RAG pipeline.

### The problem it solves

Retrieval returns more context than fits in your prompt budget. Something must be
cut. Standard compressors cut by **relevance to the query** — they keep sentences
that "look related" and drop the rest.

A sentence containing an IBAN, tucked in an annex, does not look related to
anything. It gets dropped. The LLM answers from what remains, confidently, and the
pipeline reports success. **The failure is silent.**

### What Quasar does differently

It identifies values that must not be lost — amounts, IBANs, tax IDs, dates, legal
references — and **reserves them verbatim before relevance ranking gets a vote.**
When the budget cannot hold them all, it **reports that** rather than dropping them
quietly.

### What it is not

- Not a retriever. It does not do vector search or chunking.
- Not the highest-compression tool. Token-deletion methods compress harder.
- Not the highest-accuracy tool on open-ended QA. Plain top-k ranking beats it
  there (see [Benchmarks](#10-benchmarks)).

**Use Quasar when a wrong value is an incident**, not when you want the best
answer to a fuzzy question.

---

## 2. Installation

```bash
pip install quasar-context
```

**Requirements:** Python ≥ 3.9

**Dependencies** (installed automatically):
- `sentence-transformers` ≥ 2.2.0 — embedding model for relevance scoring
- `tiktoken` ≥ 0.5.0 — exact token counting (cl100k_base)
- `numpy` ≥ 1.21.0

**First run downloads an ~80 MB embedding model** (`all-MiniLM-L6-v2`) from
Hugging Face. This happens once and is cached. See [Performance](#9-performance)
for how to avoid paying this cost on a live request.

---

## 3. Quick start

```python
from quasar import ContextOptimizer

opt = ContextOptimizer()

result = opt.optimize(
    query="What is the total due and the payment deadline?",
    context=retrieved_chunks,        # str or list[str]
    target_tokens=500,
)

result.context          # compressed text → feed this to your LLM
result.report           # what happened, and whether it was safe
```

### With the guard (the reason Quasar exists)

```python
result = opt.optimize(query, chunks, target_tokens=500)

if not result.report.faithful:
    # The budget could not hold every critical value.
    # You now KNOW this — before the LLM answers.
    logger.warning("quasar: %s", result.report.warnings)
    result = opt.optimize(query, chunks, target_tokens=1000)  # give it room

answer = llm(query, result.context)
```

### Reading the report

```python
print(result.report.summary())
# [Quasar] 63% smaller (1284->475 tok, ~$0.0024 saved) | strategy=faithful-selection | critical 7/7 OK
```

---

## 4. Architecture

### Where it sits

```
question
   ↓
retrieval  (unchanged — pgvector / Pinecone / FAISS)
   ↓
   ↓  15 chunks · 6,400 tokens  ← too big for the prompt
   ↓
QUASAR  ←── the only new step
   ↓
   ├── faithful: True  → 500 tokens, all critical values intact → LLM
   └── faithful: False → warning naming what was dropped → your handler
```

### The pipeline inside `optimize()`

**Step 0 — Deduplicate.**
Identical sentences (normalised: whitespace-collapsed, lowercased) are collapsed to
one. Retrieval with overlapping chunk windows routinely returns the same paragraph
several times; without this you pay for it several times.

This runs **before** the passthrough check. A context of eight identical sentences
may fit under budget, but shipping eight copies wastes the budget Quasar exists to
protect.

**Step 1 — Passthrough check.**
If the deduplicated context already fits `target_tokens`, return it unchanged. No
embedding cost, no risk of dropping anything.

Strategy reported as `passthrough (already within budget)` or
`dedupe-only (N duplicate sentence(s) removed)`.

**Step 2 — Detect critical spans.**
Scan the full context with the pattern set (see
[Critical value detection](#7-critical-value-detection)). Every match is a value
that must survive.

**Step 3 — Reserve critical sentences.**
Sentences containing a critical span are sorted by query relevance and claim budget
**first**. They are kept **verbatim** — never rewritten, never summarised.

> This ordering is the core design decision. Every other compressor ranks by
> relevance and takes the top N. Quasar reserves what must survive, *then* ranks.

If a critical sentence cannot fit, it is dropped **and recorded as a warning.**

**Step 4 — Fill by relevance.**
Remaining budget is filled with the most query-relevant of the remaining sentences,
scored by cosine similarity between sentence and query embeddings
(`all-MiniLM-L6-v2`, normalised). Sentences below `min_relevance` are skipped.

**Step 5 — Reassemble.**
Selected sentences are re-joined **in original document order**, not selection
order, to preserve readability.

**Step 6 — Audit.**
Every detected critical span is checked against the output text. `faithful` is
`True` only if all of them survived verbatim. Any that did not are named in
`warnings`.

---

## 5. The faithfulness contract

This is the guarantee. It has two halves, and the second matters as much as the
first.

### Half 1 — Preserve

> If the token budget can hold the critical values, they survive **character-for-character**.

Not summarised. Not paraphrased. Not rewritten. The exact string `€47,350.00`
appears in the output exactly as it appeared in the input.

This is structurally guaranteed by selection: Quasar **chooses whole sentences**.
It never generates or rewrites text, so it cannot corrupt a value the way
token-deletion compressors (which rewrite) can.

### Half 2 — Or warn

> If the budget **cannot** hold them, `report.faithful` is `False` and
> `report.warnings` names exactly which values were dropped.

**Quasar never silently drops a critical value.** This is what no other compressor
offers.

### What the contract does NOT claim

It does **not** claim the values always fit. That would be physically impossible:
if a document contains six critical values totalling 90 tokens and your budget is
25, no tool can preserve them all.

**At extreme compression, Quasar drops values too.** The difference is that it
refuses to pretend it didn't.

> **Be precise when reporting this.** Quasar detects that a critical value
> *could not fit and was dropped*. It does **not** detect corruption. The
> defensible claim is *"N critical values would have been silently dropped"* —
> not *"N corrupted values caught."*

### Tested

The contract is asserted in the test suite (`tests/test_core.py`):

| Test | Asserts |
|---|---|
| `test_critical_preserved_when_budget_allows` | Exact strings survive verbatim |
| `test_warns_when_budget_too_tight` | Dropping a value sets `faithful=False` + warnings |
| `test_never_silently_drops` | Across many budgets: `faithful=False` ⟹ warnings non-empty |
| `test_dedupes_repeats` | Duplicate critical sentences don't each consume budget |

If these fail, CI blocks the release. The publish workflow will not ship a build
where the contract is broken.

---

## 6. API reference

### `ContextOptimizer`

```python
ContextOptimizer(config: OptimizerConfig = OptimizerConfig())
```

Loads the embedding model on first use (lazy, module-level singleton — creating
multiple optimizers does not reload the model).

---

#### `optimize(query, context, target_tokens=500) → OptimizationResult`

| Parameter | Type | Description |
|---|---|---|
| `query` | `str` | The question the context must answer. Drives relevance scoring. |
| `context` | `str \| list[str]` | Retrieved context. A list is joined with newlines. |
| `target_tokens` | `int` | Token budget for the output. Default 500. |

Returns an `OptimizationResult`.

---

#### `optimized_prompt(query, context, target_tokens=500, template=...) → (str, OptimizationReport)`

Convenience wrapper that formats the compressed context into a prompt template.

```python
prompt, report = opt.optimized_prompt(query, chunks, target_tokens=500)
```

Default template:
```
Context:
{context}

Question: {query}
Answer:
```

---

### `OptimizerConfig`

```python
OptimizerConfig(
    cost_per_1k_tokens: float = 0.003,
    skip_if_under:      int   = 0,
    min_relevance:      float = 0.05,
    preserve_critical:  bool  = True,
    model_name:         str   = "all-MiniLM-L6-v2",
)
```

| Field | Description |
|---|---|
| `cost_per_1k_tokens` | Your model's input price. Used only for the savings estimate in the report. |
| `min_relevance` | Sentences scoring below this cosine similarity to the query are not selected as filler. Does not affect critical sentences. |
| `preserve_critical` | The faithfulness behaviour. **Leave this on.** Setting `False` makes Quasar a plain top-k filter. |
| `model_name` | Any sentence-transformers model. Larger models are more accurate and slower. |

---

### `OptimizationResult`

| Field | Type | Description |
|---|---|---|
| `context` | `str` | The compressed text. Feed this to your LLM. |
| `report` | `OptimizationReport` | What happened. |

---

### `OptimizationReport`

| Field | Type | Description |
|---|---|---|
| `tokens_in` | `int` | Tokens in the original context |
| `tokens_out` | `int` | Tokens in the compressed context |
| `tokens_saved` | `int` | `tokens_in - tokens_out` |
| `pct_saved` | `float` | Percentage reduction |
| `est_cost_saved_usd` | `float` | Estimated saving at `cost_per_1k_tokens` |
| `strategy` | `str` | Which path was taken (`passthrough`, `dedupe-only`, `faithful-selection`) |
| `critical_found` | `int` | Distinct critical values detected in the input |
| `critical_preserved` | `int` | How many survived into the output |
| **`faithful`** | **`bool`** | **`True` iff every critical value survived. This is the field to check.** |
| `warnings` | `list[str]` | Human-readable warnings naming dropped values |
| `optimize_ms` | `float` | Wall-clock time for this call |

```python
report.summary()   # one-line string for logs
```

---

### `find_critical(text) → list[str]`

Returns every critical span detected in `text`. Useful for testing your patterns,
and for building a shadow-mode comparison.

```python
from quasar import find_critical
find_critical("Total is €47,350.00 due 15/03/2026.")
# ['€47,350.00', '15/03/2026']
```

---

## 7. Critical value detection

### Default patterns

Detection is **regex-based**, not learned. This is deliberate: a pattern either
matches or it doesn't, which makes the behaviour auditable. A learned classifier
would be probabilistic, and "probably preserved your IBAN" is not a guarantee.

| Kind | Pattern (simplified) | Matches |
|---|---|---|
| `currency` | `[€$£¥]\s?\d[\d,.]*` | `€47,350.00` |
| `large_number` | `\d{1,3}([,.]\d{3})+` | `1,250,000` |
| `vin` | `[A-HJ-NPR-Z0-9]{17}` | `WAUZZZ8K9BA123456` |
| `iban` | `RO\d{2}[A-Z0-9]{16,}` | `RO49AAAA1B31007593840000` |
| `account` | `[A-Z]{2}\d{2}[A-Z0-9]{10,}` | Generic IBAN-like |
| `date` | `\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}` | `15/03/2026` |
| `legal_ref` | `(Order\|Article\|Regulation\|Section\|Clause\|Directive)\s+[\w/.\-]+` | `Order 1802/2014` |
| `code` | `[A-Z0-9]{6,}\-[A-Z0-9]{3,}` | `CTR-2026-00847` |
| `long_id` | `\d{6,}` | Invoice numbers |

### Extending for your domain

**You will need to.** The defaults are generic; your documents have their own
formats.

```python
from quasar import core

core._CRITICAL_PATTERNS.extend([
    (r"\bRO\d{2,10}\b",           "cui"),            # Romanian fiscal code
    (r"\bJ\d{2}/\d+/\d{4}\b",     "reg_com"),        # trade register
    (r"\bCASE-\d{6}\b",           "case_id"),
    (r"\bPO-\d{5,}\b",            "purchase_order"),
])
```

**Verify against real documents before trusting them:**

```python
from quasar import find_critical

for doc in sample_invoices:
    print(find_critical(doc))
```

If a value you care about is not in that output, Quasar will not protect it.
**Undetected means unprotected.** This is the most important thing to test during
integration.

### False positives

Over-matching is cheap: a false positive means an extra sentence is reserved, using
budget. Under-matching is expensive: a false negative means a critical value gets
treated as ordinary text and may be dropped.

**When in doubt, match more.**

---

## 8. Integration patterns

### Basic RAG

```python
from quasar import ContextOptimizer, OptimizerConfig

# Startup — create once, warm once.
_opt = ContextOptimizer(OptimizerConfig(cost_per_1k_tokens=0.003))
_opt.optimize("warmup", "Warmup sentence.", target_tokens=50)


def answer(query: str) -> str:
    chunks = vector_db.search(query, top_k=15)

    result = _opt.optimize(query, chunks, target_tokens=500)

    if not result.report.faithful:
        logger.warning("quasar_unsafe", extra={
            "query": query,
            "warnings": result.report.warnings,
        })
        result = _opt.optimize(query, chunks, target_tokens=1000)

    return llm(query, result.context)
```

### Fail-closed (high-stakes extraction)

When a wrong value is worse than no value, refuse to answer rather than answer from
degraded context.

```python
result = _opt.optimize(query, chunks, target_tokens=budget)

if not result.report.faithful:
    raise CriticalDataLost(
        f"Cannot answer safely: {result.report.warnings}"
    )

return llm(query, result.context)
```

### Escalating budget

```python
for budget in (500, 1000, 2000):
    result = _opt.optimize(query, chunks, target_tokens=budget)
    if result.report.faithful:
        break
else:
    raise CriticalDataLost(result.report.warnings)
```

### Shadow mode (measuring the benefit)

Run a plain top-k filter alongside — not in the live path — to measure what you
would have lost without Quasar. See [Monitoring](#12-monitoring-in-production).

---

## 9. Performance

| Phase | Cost |
|---|---|
| **Model load** (first call only) | 30–60 s |
| **Per call, warm** | ~10–100 ms |
| Compression compute | Embedding (cheap) — not an LLM |

### The cold start

The first `optimize()` call loads `all-MiniLM-L6-v2` and takes 30–60 seconds.
**Never let a user request pay this.** Warm the optimizer at application startup:

```python
_opt = ContextOptimizer()
_opt.optimize("warmup", "Warmup sentence.", target_tokens=50)
```

The model is a module-level singleton — additional `ContextOptimizer()` instances
do not reload it.

### Relative cost

Quasar uses **embeddings** to score relevance. LLMLingua runs a **neural language
model** to compress. Measured on LongBench, Quasar's compression step is
**8–13× faster**.

In context: a ~50 ms compression step sits in front of an LLM call that typically
takes 500–5000 ms. **Compression is not your bottleneck.** It is a rounding error
against generation time, and it *reduces* generation time by shortening the prompt.

---

## 10. Benchmarks

Measured on **LongBench** (narrativeqa, qasper, hotpotqa), N=30 per task, a real
LLM judge, token-F1 scoring. Budgets 200 / 500 / 1000.

### Results

| Matchup | Result |
|---|---|
| vs **truncation** (keep first N tokens) | **9–0 win** |
| vs **LLMLingua** (Microsoft) | **6–0 win** on accuracy, **8–13× faster** |
| vs **LangChain `EmbeddingsFilter`** (top-k) | **3–6 loss** |
| Long documents (17k+ words) | Quasar handles them; **LLMLingua cannot process them at all** |

### Reading this honestly

**Quasar loses to a plain top-k embedding filter on QA accuracy.** For open-ended
questions ("what's the gist of this report?"), simple relevance ranking gives
slightly better answers. That is stated here rather than hidden, because it is true
and because you would find it in ten minutes anyway.

**Those benchmarks measure answer relevance. None of them measure whether the
invoice number survived.** A top-k filter will drop a low-relevance sentence
containing an IBAN and never mention it. That failure is invisible to token-F1 and
fatal in accounts payable.

**Choose accordingly:**

| Your situation | Use |
|---|---|
| General chatbot, open-ended Q&A | A top-k filter. It's simpler and scores better. |
| Maximum compression, text can be rewritten | LLMLingua. |
| **Exact values must survive; a wrong number is an incident** | **Quasar.** |

---

## 11. Limitations

**Stated plainly. Read these before integrating.**

### It cannot recover what retrieval missed

If the chunk containing the IBAN was never retrieved, Quasar cannot preserve it.
**Fix retrieval first.** Quasar protects the values in the chunks you *did* get.

Retrieval failure and compression failure are different problems. Quasar solves the
second.

### It cannot fit N values into a budget smaller than N

At extreme compression, critical values *will* be dropped. Quasar reports it; it
does not perform miracles.

At very tight budgets, Quasar and a top-k filter may drop the *same number* of
values — the difference is only that Quasar tells you. That is a real advantage,
but a weaker one than preservation. Do not oversell it.

### Detection is regex-based

A value in an unanticipated format is **not detected**, and therefore **not
protected**. Test `find_critical()` against your real documents. Undetected means
unprotected.

### It does not detect corruption

Quasar knows a value *did not fit and was dropped*. It has no way to know a value
was *altered* — nothing upstream of it rewrites text. Be precise when reporting:
*"would have been silently dropped"*, not *"corrupted"*.

### Sentence-level granularity

Selection operates on sentences. A very long sentence containing one critical value
consumes its full token cost. Quasar cannot keep "half a sentence".

### Tests require a model download

The test suite downloads `all-MiniLM-L6-v2`. In sandboxed CI without network
access, two tests will fail. This is an environment issue, not a code failure —
but it means contributors cannot run the full suite offline.

---

## 12. Monitoring in production

Quasar returns a full report per call. **Log it to your own monitoring — the
library sends nothing anywhere.**

### The metrics that matter

```python
result = opt.optimize(query, chunks, target_tokens=500)

# The headline safety metric
metrics.increment("quasar.unsafe", 0 if result.report.faithful else 1)

# The cost metric
metrics.gauge("quasar.tokens_saved", result.report.tokens_saved)

# The forensic record — WHICH values were at risk
if not result.report.faithful:
    logger.warning("quasar_critical_dropped", extra={
        "query":              query,
        "warnings":           result.report.warnings,
        "critical_found":     result.report.critical_found,
        "critical_preserved": result.report.critical_preserved,
    })
```

An alert on `quasar.unsafe` tells you your token budget is too tight for your
documents — **before** a customer finds a wrong number.

### Shadow mode — measuring what Quasar prevented

To quantify the benefit, run a plain top-k filter in parallel (off the critical
path) and record what it *would have* dropped.

```python
from quasar import find_critical

shadow_ctx     = plain_topk(query, chunks, budget)   # your baseline
all_critical   = set(find_critical("\n".join(chunks)))
would_be_lost  = [v for v in all_critical if v not in shadow_ctx]

audit.insert(
    faithful          = result.report.faithful,
    critical_found    = result.report.critical_found,
    critical_kept     = result.report.critical_preserved,
    baseline_dropped  = would_be_lost,     # the number that matters
)
```

**Keep the comparison fair:** same sentence splitting, same embedding model, same
budget. A tilted baseline produces a number you cannot defend.

**The claim this produces:**

> *"Over N production parses, plain relevance ranking would have silently dropped
> M critical values — invoice totals, IBANs, deadlines. Quasar preserved P and
> flagged Q as unsafe."*

**If the baseline drops nothing on your documents, report that.** It means your
critical values live in high-relevance sentences, top-k finds them anyway, and
Quasar's guarantee buys you little. That is a valid finding. Do not massage it.

---

## 13. Troubleshooting

**First call takes 60 seconds**
The embedding model is downloading. One-time. Warm at startup
(see [Performance](#9-performance)).

**`faithful` is always `False`**
Your budget is smaller than the critical content requires. Check
`report.critical_found` and raise `target_tokens`. If a document has 12 critical
values, a 100-token budget cannot hold them.

**A value I care about is being dropped**
It is probably not being detected. Check:
```python
from quasar import find_critical
find_critical(your_document)
```
If it is not in that list, add a pattern
(see [Critical value detection](#7-critical-value-detection)).

**Output is identical at different budgets**
The deduplicated content already fits both budgets. Quasar keeps what is relevant
and stops — it does not pad to fill the budget.

**Tests fail with a Hugging Face connection error**
The test suite needs the model. Pre-download it:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**Compression is slower than expected**
Confirm the model is warm and that you are not creating a new `ContextOptimizer`
per request. The model is a singleton, but object churn still costs.

---

## Appendix — design rationale

### Why reserve-then-rank, rather than rank-with-a-bonus?

An earlier design scored sentences with a combined function (relevance + a
criticality bonus) and took the top N. It was benchmarked and **rejected**: a
critical sentence with low query relevance could still lose to several highly
relevant ordinary sentences. A bonus is a preference; **reservation is a
guarantee.**

### Why regex, not a learned detector?

A learned classifier is probabilistic. "Probably preserved your IBAN" is not a
contract. A regex either matches or it does not, which makes the behaviour
auditable and testable — and the failure mode (an unmatched format) is *visible*
and fixable, rather than a silent low-confidence score.

### Why no redundancy penalty?

An earlier version penalised sentences similar to already-selected ones (a
"repulsion" term). Benchmarked against real data, **it actively hurt accuracy** —
it avoided clusters of related answer-bearing sentences, which are exactly what
multi-fact questions need. It was removed.

The shipped product is deliberately simple: detect, reserve, rank, audit. Every
mechanism that did not survive a benchmark was cut.
