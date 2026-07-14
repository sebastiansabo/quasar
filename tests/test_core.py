"""
Tests for Quasar. The critical ones assert the FAITHFULNESS CONTRACT:
  - critical values survive verbatim when budget allows
  - when budget is too tight, the report says so (never silent)
If these fail, the product's only differentiator is broken.
"""
import pytest
from quasar import ContextOptimizer, OptimizerConfig, find_critical


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def opt():
    return ContextOptimizer(OptimizerConfig(cost_per_1k_tokens=0.003))


CRITICAL_CHUNKS = [
    "The quarterly business review covered several operational topics.",
    "The invoice total amount due is €47,350.00 payable by 15/03/2026.",
    "Market conditions have remained generally stable this period.",
    "Payment should be remitted to IBAN RO49AAAA1B31007593840000 promptly.",
    "The team continues to monitor performance metrics across departments.",
    "The applicable regulation is Order 1802/2014 governing this deal.",
    "Documentation has been updated to reflect procedural changes made.",
]
QUERY = "What is the total amount due and the payment deadline?"


# ---------------------------------------------------------------- detection
def test_finds_currency():
    assert any("47,350" in c for c in find_critical("Total is €47,350.00 due now."))

def test_finds_iban():
    hits = find_critical("Send to RO49AAAA1B31007593840000 today.")
    assert any("RO49AAAA" in h for h in hits)

def test_finds_date():
    assert find_critical("Due by 15/03/2026 without fail.")

def test_ignores_plain_prose():
    assert find_critical("The weather today is mild and pleasant.") == []


# ---------------------------------------------------------------- the contract
def test_critical_preserved_when_budget_allows(opt):
    """THE core claim: with room, every critical value survives verbatim."""
    res = opt.optimize(QUERY, CRITICAL_CHUNKS, target_tokens=300)
    assert res.report.faithful, f"critical data lost: {res.report.warnings}"
    assert "€47,350.00" in res.context
    assert "RO49AAAA1B31007593840000" in res.context
    assert "15/03/2026" in res.context
    assert res.report.critical_preserved == res.report.critical_found

def test_warns_when_budget_too_tight(opt):
    """THE honesty claim: if it can't fit critical data, it MUST say so."""
    res = opt.optimize(QUERY, CRITICAL_CHUNKS, target_tokens=15)
    if not res.report.faithful:
        assert res.report.warnings, "dropped critical data but issued NO warning"
        assert res.report.critical_preserved < res.report.critical_found

def test_never_silently_drops(opt):
    """faithful==False must always be accompanied by warnings."""
    for budget in (10, 20, 30, 50, 80, 200, 400):
        res = opt.optimize(QUERY, CRITICAL_CHUNKS, target_tokens=budget)
        if not res.report.faithful:
            assert res.report.warnings, f"silent drop at budget={budget}"


# ---------------------------------------------------------------- behavior
def test_passthrough_when_already_small(opt):
    res = opt.optimize(QUERY, "Short context.", target_tokens=500)
    assert "passthrough" in res.report.strategy
    assert res.report.tokens_saved == 0
    assert res.report.faithful

def test_actually_compresses(opt):
    big = CRITICAL_CHUNKS * 6
    res = opt.optimize(QUERY, big, target_tokens=200)
    assert res.report.tokens_out <= 200
    assert res.report.tokens_saved > 0
    assert res.report.pct_saved > 0

def test_dedupes_repeats(opt):
    """Duplicate sentences must not each consume budget."""
    dupes = ["The invoice total amount due is €47,350.00 payable soon."] * 8
    res = opt.optimize(QUERY, dupes, target_tokens=100)
    assert res.context.count("€47,350.00") == 1, "kept duplicate critical sentences"

def test_accepts_string_or_list(opt):
    as_list = opt.optimize(QUERY, CRITICAL_CHUNKS, target_tokens=200)
    as_str = opt.optimize(QUERY, " ".join(CRITICAL_CHUNKS), target_tokens=200)
    assert as_list.report.tokens_out == as_str.report.tokens_out


# ---------------------------------------------------------------- report
def test_report_is_complete(opt):
    res = opt.optimize(QUERY, CRITICAL_CHUNKS * 4, target_tokens=150)
    r = res.report
    assert r.tokens_in > r.tokens_out
    assert r.tokens_saved == r.tokens_in - r.tokens_out
    assert 0 <= r.pct_saved <= 100
    assert r.est_cost_saved_usd >= 0
    assert isinstance(r.faithful, bool)
    assert r.summary()

def test_empty_context_does_not_crash(opt):
    res = opt.optimize(QUERY, "", target_tokens=100)
    assert res.context == "" or res.report.faithful
