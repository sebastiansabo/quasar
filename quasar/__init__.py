"""
Quasar — faithful context optimization for RAG apps.

Compress retrieved context before your LLM call. Critical values
(prices, IBANs, dates, IDs, legal refs) are preserved verbatim when the
budget allows — and you get an explicit warning when it can't, instead of
silent corruption.

Quick start:
    from quasar import ContextOptimizer

    opt = ContextOptimizer()
    result = opt.optimize(query, retrieved_chunks, target_tokens=500)

    result.context          # compressed context -> feed to your LLM
    result.report.faithful  # bool: did every critical value survive?
    print(result.report.summary())
"""

from .core import (
    ContextOptimizer,
    OptimizerConfig,
    OptimizationResult,
    OptimizationReport,
    find_critical,
)

__version__ = "0.1.0"
__all__ = [
    "ContextOptimizer",
    "OptimizerConfig",
    "OptimizationResult",
    "OptimizationReport",
    "find_critical",
]
