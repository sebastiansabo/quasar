# Quasar

**Faithful context optimization for RAG apps.**

Compress retrieved context before your LLM call — without silently corrupting the
values that matter. Prices, IBANs, dates, invoice numbers, legal references and
IDs are preserved **verbatim** when the budget allows. When it doesn't, Quasar
**tells you** instead of quietly dropping them.

📖 **[Full technical documentation](DOCUMENTATION.md)** — architecture, API
reference, the faithfulness contract, benchmarks, and limitations.

```python
from quasar import ContextOptimizer

opt = ContextOptimizer()
result = opt.optimize(query, retrieved_chunks, target_tokens=500)

llm_answer(query, result.context)      # feed the compressed context
print(result.report.summary())
# [Quasar] 63% smaller (1284->475 tok, ~$0.0024 saved) | critical 7/7 OK
```

---

## Why this exists

Every context compressor optimizes for one thing: fewer tokens. None of them tell
you what they destroyed on the way.

If your RAG app retrieves an invoice and the compressor drops or mangles
`€47,350.00`, your LLM confidently answers with the wrong number and **you never
find out**. That's fine for a chatbot. It is not fine for finance, legal,
medical, compliance, or code.

Quasar makes one guarantee no other compressor makes:

> **It never silently corrupts your critical data.**
> Critical values survive verbatim, or you get an explicit warning.

---

## What it does

1. **Detects critical spans** — currency, IBANs, dates, VINs, long IDs, legal
   references, contract codes — by pattern.
2. **Reserves them verbatim.** Critical sentences claim budget first and are
   never rewritten. (Token-deletion compressors *rewrite* text — that's how they
   corrupt exact values.)
3. **Fills the remaining budget by relevance**, using embedding similarity to the
   query. Filler is dropped.
4. **Dedupes** repeated sentences so budget isn't wasted on copies.
5. **Audits and reports.** Every call returns tokens saved, cost saved, and a
   faithfulness status — with warnings naming any critical value it couldn't fit.

---

## Honest benchmarks

Measured on LongBench (narrativeqa / qasper / hotpotqa, N=30 per task), real LLM
judge, token-F1 scoring.

**Where Quasar wins:**

| Matchup | Result |
|---|---|
| vs **truncation** | **9–0** — wins at every task and budget |
| vs **LLMLingua** | **6–0** on accuracy, **8–13× faster** to compress |
| Long contexts | Handles 17k+ word documents LLMLingua **cannot process at all** |

LLMLingua runs a neural model to compress. Quasar runs embeddings. Same or better
accuracy, a fraction of the compute.

**Where Quasar does not win — stated plainly:**

Against a **pure top-k embedding filter** (e.g. LangChain's `EmbeddingsFilter`),
Quasar goes **3–6** on raw QA accuracy. On pure question-answering, simple
relevance ranking is competitive or better.

**So Quasar is not the accuracy leader on QA, and doesn't claim to be.** Its value
is the axis those benchmarks don't measure: *faithfulness*. A top-k filter has no
concept of critical data — it drops a low-relevance sentence containing your
invoice number without hesitation, and never tells you. Quasar reserves it, and
warns when it can't.

**Use Quasar when correctness of exact values matters more than the last 2% of
retrieval F1.** If you're building a general chatbot and only care about QA
accuracy, use a top-k filter — it's simpler and we'll say so.

---

## Install

```bash
pip install quasar-context
```

## Usage

### Basic

```python
from quasar import ContextOptimizer

opt = ContextOptimizer()          # loads the embedding model once
result = opt.optimize(
    query="What is the total due and the deadline?",
    context=retrieved_chunks,     # str or list[str]
    target_tokens=500,
)

result.context                    # compressed text for your LLM
result.report.tokens_saved        # int
result.report.faithful            # bool — did all critical values survive?
result.report.warnings            # list[str] — what got dropped and why
```

### Guarding a production call

```python
result = opt.optimize(query, chunks, target_tokens=500)

if not result.report.faithful:
    # budget too tight to hold every critical value — your call:
    logger.warning("Quasar: %s", result.report.warnings)
    result = opt.optimize(query, chunks, target_tokens=1000)   # give it room

answer = llm(query, result.context)
```

### Configuration

```python
from quasar import ContextOptimizer, OptimizerConfig

opt = ContextOptimizer(OptimizerConfig(
    cost_per_1k_tokens=0.003,   # your model's price, for the savings report
    preserve_critical=True,     # the faithfulness behavior (default on)
    min_relevance=0.05,         # drop sentences below this query relevance
    model_name="all-MiniLM-L6-v2",
))
```

### Custom critical patterns

Your domain has its own critical formats. Add them:

```python
from quasar import core
core._CRITICAL_PATTERNS.append((r"\bCASE-\d{6}\b", "case_id"))
```

---

## Performance

- **~10–100 ms per call** after warm-up (embedding + greedy selection).
- **First call loads the model** (~30–60s, one time). Warm it at startup:
  ```python
  opt = ContextOptimizer()
  opt.optimize("warmup", "warmup text.", target_tokens=50)
  ```
- **8–13× faster than LLMLingua**, which runs a full neural model to compress.

---

## What Quasar is not

- **Not the best compressor by ratio.** Token-deletion methods compress harder.
  They also rewrite your text.
- **Not an accuracy leader on pure QA.** A top-k filter matches or beats it there.
- **Not a guarantee against physics.** If your critical content is larger than
  your token budget, it cannot all fit. Quasar preserves what it can *and tells
  you what it couldn't* — that's the honest contract.

---

## License

MIT.
