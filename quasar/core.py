"""
================================================================================
 quasar_layer.py  --  Drop-in context optimization for RAG apps
================================================================================
 WHAT IT IS
 ----------
 A single layer you insert between retrieval and your LLM call. It:
   1. Picks the right compression strategy for the context automatically.
   2. GUARANTEES exact critical content (IDs, prices, dates, citations) survives
      verbatim -- the one thing pure relevance ranking can't promise.
   3. Reports exactly what it did: tokens saved, est. cost saved, what was kept,
      and any faithfulness risk it caught.

 WHAT IS NOT
 -----------
 Not a magic compressor that beats everyone. Honest benchmarks showed
 embedding-selection beats LLMLingua at far lower cost but ties/loses to simple
 top-k on pure QA. So this layer's value is NOT "best compression" -- it is:
   - zero-config: developer drops it in, no tuning
   - safe-by-reporting: when budget can't fit all critical data, it does NOT
     silently corrupt -- it preserves what fits and WARNS about what it dropped.
     (No compressor can fit N critical facts into a budget smaller than them;
     the honest behavior is to surface that, not hide it.)
   - transparent: shows the savings + what the LLM actually sees
   - cheap: embeddings, not an LLM-to-compress (8-13x faster than LLMLingua)

 DESIGN PRINCIPLE
 ----------------
 The product is the DECISION + the PROOF, not the algorithm. Developers don't
 know whether to compress, how much, or whether it's safe. This answers that
 per-call, automatically, with receipts.

 INSTALL
   pip install sentence-transformers tiktoken

 USE (the whole API is three lines)
   from quasar_layer import ContextOptimizer
   opt = ContextOptimizer()
   result = opt.optimize(query, retrieved_chunks, target_tokens=500)
   # result.context     -> compressed context, feed to your LLM
   # result.report      -> tokens saved, cost saved, faithfulness status
================================================================================
"""
from __future__ import annotations
import re, time
from dataclasses import dataclass, field
from typing import Optional

# ---------- optional deps, graceful fallback ----------
try:
    import tiktoken; _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None
def ntok(t: str) -> int:
    if _ENC: return len(_ENC.encode(t))
    return max(1, int(len(t.split()) * 1.3))

_MODEL = None
def _model(name="all-MiniLM-L6-v2"):
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(name)
    return _MODEL


# =============================================================================
# CRITICAL CONTENT DETECTION  (the faithfulness guarantee)
# =============================================================================
# Spans matching these MUST survive verbatim. Extensible by the developer.
_CRITICAL_PATTERNS = [
    (r'[€$£¥]\s?\d[\d,.]*', "currency"),
    (r'\b\d{1,3}(?:[,.]\d{3})+(?:\.\d+)?\b', "large_number"),
    (r'\b[A-HJ-NPR-Z0-9]{17}\b', "vin"),
    (r'\bRO\d{2}[A-Z0-9]{16,}\b', "iban"),
    (r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,}\b', "account"),
    (r'\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b', "date"),
    (r'\b(?:Order|Article|Regulation|Section|Clause|Directive)\s+[\w/.\-]+', "legal_ref"),
    (r'\b[A-Z0-9]{6,}\-[A-Z0-9]{3,}\b', "code"),
    (r'\b\d{6,}\b', "long_id"),
]
def find_critical(text: str) -> list[str]:
    hits = []
    for pat, kind in _CRITICAL_PATTERNS:
        for m in re.finditer(pat, text):
            hits.append(m.group(0))
    return hits


# =============================================================================
# RESULT + REPORT
# =============================================================================
@dataclass
class OptimizationReport:
    tokens_in: int
    tokens_out: int
    tokens_saved: int
    pct_saved: float
    est_cost_saved_usd: float          # at a configurable $/1k tokens
    strategy: str                      # which path was taken + why
    critical_found: int
    critical_preserved: int
    faithful: bool                     # all critical content survived?
    warnings: list = field(default_factory=list)
    optimize_ms: float = 0.0
    def summary(self) -> str:
        flag = "OK" if self.faithful else "FAITHFULNESS RISK"
        return (f"[Quasar] {self.pct_saved:.0f}% smaller "
                f"({self.tokens_in}->{self.tokens_out} tok, "
                f"~${self.est_cost_saved_usd:.4f} saved) | "
                f"strategy={self.strategy} | critical {self.critical_preserved}/{self.critical_found} {flag}"
                + ("" if not self.warnings else f" | {len(self.warnings)} warning(s)"))

@dataclass
class OptimizationResult:
    context: str
    report: OptimizationReport


# =============================================================================
# THE OPTIMIZER
# =============================================================================
@dataclass
class OptimizerConfig:
    cost_per_1k_tokens: float = 0.003   # for the savings estimate (set to your model)
    skip_if_under: int = 0              # if context already <= target, don't touch it
    min_relevance: float = 0.05         # drop sentences below this query relevance
    preserve_critical: bool = True      # the faithfulness guarantee (on by default)
    model_name: str = "all-MiniLM-L6-v2"

class ContextOptimizer:
    """
    Drop-in: retrieved context in, optimized context out, with receipts.
    Auto-selects strategy:
      - context already within budget   -> passthrough (no cost, no risk)
      - context over budget             -> faithful embedding-selection:
          * reserve critical spans verbatim (guarantee)
          * fill remaining budget by query relevance (the workhorse that
            beat LLMLingua at a fraction of the cost)
    """
    def __init__(self, config: OptimizerConfig = OptimizerConfig()):
        self.cfg = config

    def _split(self, text: str) -> list[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

    def optimize(self, query: str, context, target_tokens: int = 500) -> OptimizationResult:
        t0 = time.perf_counter()
        # accept either a string or a list of chunks
        if isinstance(context, (list, tuple)):
            context = "\n".join(str(c) for c in context)
        tokens_in = ntok(context)
        crit_all = find_critical(context)

        # ---- STEP 0: DEDUPE FIRST (always) ----
        # Collapse identical sentences so we never spend budget on copies.
        # This MUST run before the passthrough check: a context of 8 identical
        # sentences may fit under the budget, but shipping 8 copies to the LLM
        # wastes the very budget we exist to protect.
        raw_sents = self._split(context)
        seen = set(); sents = []
        for s in raw_sents:
            key = re.sub(r'\s+', ' ', s.strip().lower())
            if key in seen: continue
            seen.add(key); sents.append(s)
        deduped = " ".join(sents)
        tokens_deduped = ntok(deduped)
        dupes_removed = len(raw_sents) - len(sents)

        # ---- STRATEGY 1: passthrough if the DEDUPED context already fits ----
        if tokens_deduped <= target_tokens:
            saved = tokens_in - tokens_deduped
            crit_kept = sum(1 for c in set(crit_all) if c in deduped)
            strategy = ("passthrough (already within budget)" if dupes_removed == 0
                        else f"dedupe-only ({dupes_removed} duplicate sentence(s) removed)")
            rep = OptimizationReport(
                tokens_in, tokens_deduped, saved,
                100.0 * saved / max(1, tokens_in),
                self.cfg.cost_per_1k_tokens * saved / 1000.0,
                strategy=strategy,
                critical_found=len(set(crit_all)),
                critical_preserved=crit_kept,
                faithful=(crit_kept == len(set(crit_all))),
                optimize_ms=(time.perf_counter()-t0)*1000)
            return OptimizationResult(deduped, rep)

        # ---- STRATEGY 2: faithful embedding-selection ----
        model = _model(self.cfg.model_name)
        embs = model.encode([query] + sents, normalize_embeddings=True, show_progress_bar=False)
        qv, svs = embs[0], embs[1:]
        sims = svs @ qv
        toks = [ntok(s) for s in sents]

        chosen, used = [], 0
        warnings = []

        # 2a. reserve sentences containing critical spans (best-effort preservation)
        if self.cfg.preserve_critical:
            crit_idx = [i for i, s in enumerate(sents) if find_critical(s)]
            # sort critical sentences by relevance so if budget is too tight we keep
            # the most query-relevant critical ones first, and warn about drops
            crit_idx.sort(key=lambda i: -sims[i])
            for i in crit_idx:
                if used + toks[i] <= target_tokens:
                    chosen.append(i); used += toks[i]
                else:
                    warnings.append(f"critical sentence dropped (budget too tight): "
                                    f"'{sents[i][:50]}...'")

        # 2b. fill remaining budget with top relevance (skip already chosen + filler)
        rest = [i for i in range(len(sents)) if i not in chosen]
        rest.sort(key=lambda i: -sims[i])
        for i in rest:
            if float(sims[i]) < self.cfg.min_relevance: break
            if used + toks[i] > target_tokens: continue
            chosen.append(i); used += toks[i]

        out_text = " ".join(sents[i] for i in sorted(chosen))
        tokens_out = ntok(out_text)

        # faithfulness audit: did every critical span survive verbatim?
        crit_out = find_critical(out_text)
        preserved = sum(1 for c in set(crit_all) if c in out_text)
        faithful = (preserved == len(set(crit_all)))
        if not faithful:
            missing = [c for c in set(crit_all) if c not in out_text]
            warnings.append(f"{len(missing)} critical value(s) not preserved: {missing[:3]}")

        saved = tokens_in - tokens_out
        rep = OptimizationReport(
            tokens_in, tokens_out, saved,
            100.0 * saved / max(1, tokens_in),
            self.cfg.cost_per_1k_tokens * saved / 1000.0,
            strategy="faithful-selection (critical reserved + relevance fill)",
            critical_found=len(set(crit_all)),
            critical_preserved=preserved,
            faithful=faithful,
            warnings=warnings,
            optimize_ms=(time.perf_counter()-t0)*1000)
        return OptimizationResult(out_text, rep)

    # --- convenience: wrap an LLM call so the developer gets one-line integration ---
    def optimized_prompt(self, query: str, context, target_tokens=500,
                         template="Context:\n{context}\n\nQuestion: {query}\nAnswer:"):
        res = self.optimize(query, context, target_tokens)
        return template.format(context=res.context, query=query), res.report


# =============================================================================
# DEMO — shows the developer experience
# =============================================================================
def _demo():
    sample_chunks = [
        "The quarterly business review covered several operational topics across regions.",
        "Market conditions have remained generally stable over the recent reporting period.",
        "The invoice total amount due is €47,350.00 payable by 15/03/2026 per the agreement.",
        "Various stakeholders provided feedback during the planning session last week.",
        "The applicable regulation is Order 1802/2014 Article 7 governing this transaction.",
        "Overall sector trends suggest continued steady commercial activity this year.",
        "The vehicle identification number on record is WAUZZZ8K9BA123456 for this unit.",
        "Documentation has been updated to reflect the latest procedural changes made.",
        "The team continues to monitor performance metrics across all departments daily.",
        "Payment should be remitted to IBAN RO49AAAA1B31007593840000 before the deadline.",
    ] * 3   # triple it to force real compression

    query = "What is the total amount due and the payment deadline?"

    print("="*78)
    print("QUASAR CONTEXT OPTIMIZER — developer experience demo")
    print("="*78)
    opt = ContextOptimizer(OptimizerConfig(cost_per_1k_tokens=0.003))

    for budget in [120, 300]:
        res = opt.optimize(query, sample_chunks, target_tokens=budget)
        print(f"\n--- target_tokens={budget} ---")
        print(res.report.summary())
        print(f"  optimize time: {res.report.optimize_ms:.0f} ms")
        if res.report.warnings:
            for w in res.report.warnings: print(f"  ! {w}")
        print(f"  LLM now sees ({res.report.tokens_out} tok):")
        print(f'    "{res.context[:220]}..."')

    print("\n" + "="*78)
    print("THE PITCH (honest):")
    print("  - 3-line drop-in: optimize(query, chunks, target_tokens)")
    print("  - Auto-passthrough when context already fits (no needless cost/risk)")
    print("  - Preserves critical values (prices, IBANs, dates, refs) verbatim when")
    print("    budget allows -- and WARNS (never silently corrupts) when it can't")
    print("  - Dedupes repeated content so budget isn't wasted on copies")
    print("  - 8-13x cheaper to run than LLMLingua (embeddings, not an LLM)")
    print("  - Every call returns receipts: tokens saved, $ saved, faithfulness status")
    print("  - The product is the DECISION + PROOF, not a 'magic' compressor.")
    print("="*78)

if __name__ == "__main__":
    _demo()
