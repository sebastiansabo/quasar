"""
Quasar in a real RAG pipeline — the integration pattern.

Run:  python examples/basic_rag.py
"""
from quasar import ContextOptimizer, OptimizerConfig

# 1. Create the optimizer once, at startup (loads the embedding model).
opt = ContextOptimizer(OptimizerConfig(
    cost_per_1k_tokens=0.003,     # your LLM's input price, for the savings report
))

# Warm it so the first real request doesn't pay the model-load cost.
opt.optimize("warmup", "Warmup sentence.", target_tokens=50)


def answer_question(query: str, retrieved_chunks: list[str]) -> str:
    """Your RAG handler. Quasar sits between retrieval and the LLM."""

    # 2. Compress the retrieved context.
    result = opt.optimize(query, retrieved_chunks, target_tokens=500)

    # 3. THE GUARD — this is why Quasar exists.
    #    If the budget was too tight to hold every critical value, you find out
    #    HERE, not when a customer notices a wrong invoice number.
    if not result.report.faithful:
        print(f"[warn] critical data at risk: {result.report.warnings}")
        # give it more room and retry rather than answer from corrupted context
        result = opt.optimize(query, retrieved_chunks, target_tokens=1000)

    print(result.report.summary())

    # 4. Feed the compressed context to your LLM.
    # return your_llm(query, result.context)
    return result.context


if __name__ == "__main__":
    chunks = [
        "The quarterly business review covered several operational topics.",
        "The invoice total amount due is €47,350.00 payable by 15/03/2026.",
        "Market conditions have remained generally stable this period.",
        "Payment should be remitted to IBAN RO49AAAA1B31007593840000 promptly.",
        "The team continues to monitor performance metrics across departments.",
        "The applicable regulation is Order 1802/2014 Article 7 governing this.",
        "Documentation has been updated to reflect procedural changes made.",
        "Overall sector trends suggest continued steady commercial activity.",
    ] * 4  # simulate a large retrieval

    context = answer_question(
        "What is the total amount due and the payment deadline?", chunks
    )
    print("\n--- context sent to LLM ---")
    print(context[:300], "...")
