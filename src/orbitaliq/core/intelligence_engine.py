"""Deterministic, explainable competitive-momentum scoring.

The business rule set here is intentionally transparent (a weighted,
capped, auditable formula) rather than a black-box model — an analyst
citing this score in a client deliverable needs to be able to justify it
line by line. The LLM (see ``report_agent``) is used only to *narrate*
this already-computed, deterministic result, never to invent the number
itself. This separation (deterministic decision, generative explanation)
is a standard, defensible pattern for AI-assisted research and diligence
workflows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from orbitaliq.core.data_quality_state import is_trusted
from orbitaliq.data.sec_edgar_client import FinancialSignal
from orbitaliq.nvidia.vision_model import VisionFindings

# Weight of the financial (SEC EDGAR) composite vs. the satellite
# (bi-temporal change) composite in the final momentum score. Kept as
# named constants (not magic numbers) so the weighting policy can be
# reviewed and versioned independently of code.
FINANCIAL_WEIGHT = 0.6
SATELLITE_WEIGHT = 0.4
# Max bonus points awarded when financial and satellite evidence agree.
CONVERGENCE_BONUS_MAX = 8.0
# Both composites must clear this bar (0-100 scale) before a convergence
# bonus applies -- otherwise one weak, noisy signal could drag the other
# up rather than genuinely corroborate it.
CONVERGENCE_THRESHOLD = 50.0

MOMENTUM_LEVEL_THRESHOLDS: dict[str, tuple[float, float]] = {
    "STABLE": (0, 30),
    "EMERGING": (30, 55),
    "STRONG": (55, 75),
    "AGGRESSIVE_EXPANSION": (75, 100.0001),
}


class MomentumLevel(str, Enum):
    STABLE = "STABLE"
    EMERGING = "EMERGING"
    STRONG = "STRONG"
    AGGRESSIVE_EXPANSION = "AGGRESSIVE_EXPANSION"
    # Not a scored level -- see _weighted_composite()/compute_momentum_
    # assessment() below. Returned only when every underlying financial AND
    # satellite signal was a live-mode fallback/unavailable, so there is
    # nothing legitimate left to weigh. Never produced by
    # classify_momentum_level(), and never mixed with a "real" score.
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class MomentumAssessmentResult:
    momentum_score: float
    momentum_level: MomentumLevel
    signal_breakdown: dict[str, float]
    recommended_actions: list[str]
    # Signal names dropped from the score because their live fetch failed
    # and only a labeled synthetic fallback (or nothing at all) was
    # available -- always empty when data_sources wasn't passed to
    # compute_momentum_assessment() (offline mode, or a direct unit-test
    # call), since nothing is excluded there.
    excluded_signals: list[str] = field(default_factory=list)
    # The three real, already-computed terms of the composite formula
    # (financial_composite: mean of trusted financial signal values 0-100;
    # satellite_composite: mean of trusted satellite signal values 0-100;
    # convergence_bonus: the agreement bonus described in
    # compute_momentum_assessment()'s docstring) -- exposed here, not
    # recomputed elsewhere, specifically so core/score_explainability.py
    # can build an accurate "Why this score?" breakdown from the exact
    # values the formula actually used, never a reconstruction. None for a
    # composite that was fully excluded (nothing trusted on that side).
    financial_composite: float | None = None
    satellite_composite: float | None = None
    convergence_bonus: float = 0.0

    def as_dict(self) -> dict:
        return {
            "momentum_score": round(self.momentum_score, 2),
            "momentum_level": self.momentum_level.value,
            "signal_breakdown": {k: round(v, 2) for k, v in self.signal_breakdown.items()},
            "recommended_actions": self.recommended_actions,
            "excluded_signals": self.excluded_signals,
            "financial_composite": round(self.financial_composite, 2) if self.financial_composite is not None else None,
            "satellite_composite": round(self.satellite_composite, 2) if self.satellite_composite is not None else None,
            "convergence_bonus": round(self.convergence_bonus, 2),
        }


def classify_momentum_level(score: float) -> MomentumLevel:
    for level_name, (low, high) in MOMENTUM_LEVEL_THRESHOLDS.items():
        if low <= score < high:
            return MomentumLevel(level_name)
    return MomentumLevel.AGGRESSIVE_EXPANSION if score >= 75 else MomentumLevel.STABLE


def _is_trusted_source(source: str | None) -> bool:
    """Mirrors ``core/data_quality_state.py``'s canonical 5-state contract
    exactly (via ``is_trusted()``), so the composite score, the Data
    Quality Center, and the DataQualityState vocabulary can never disagree
    about what was "real" for a given assessment.

    A signal is trusted (counted toward the score) when it classifies as
    ``LIVE``, ``CACHED_REAL``, or ``DERIVED`` — never when it classifies as
    ``INSUFFICIENT_DATA`` or ``ERROR``. ``source is None`` (no provenance
    passed at all, e.g. a direct unit-test call to this function) also
    trusts the signal, preserving this function's original behavior for
    every caller that predates ``data_sources`` support.
    """
    return is_trusted(source)


def _signal_value(signal: FinancialSignal) -> float | None:
    """``None`` when the underlying ``FinancialSignal`` is itself
    ``INSUFFICIENT_DATA`` (a live-mode SEC EDGAR failure -- see
    ``data/sec_edgar_client.py::_insufficient_financial_signal``). Guarding
    here, before the ``100 *`` multiply, is what makes it impossible for a
    missing input to silently become zero.
    """
    return 100 * signal.value if signal.value is not None else None


def compute_momentum_assessment(
    *,
    financial_signals: dict[str, FinancialSignal],
    vision_findings: VisionFindings,
    data_sources: dict[str, str] | None = None,
) -> MomentumAssessmentResult:
    """Fuse SEC EDGAR financial-growth signals and bi-temporal satellite
    change-detection findings into a single 0-100 competitive-momentum
    score with a per-signal breakdown.

    ``data_sources`` (the same per-signal provenance map the Ingestion
    Agent already produces and the Data Quality Center already reports —
    see ``core/data_quality.py``) is optional and additive: omitting it
    reproduces this function's original behavior exactly (every signal
    trusted, nothing excluded) so every pre-existing caller/test is
    unaffected. When it *is* passed (the real pipeline always passes it —
    see ``agents/orchestrator.py``), any signal whose live fetch genuinely
    failed and fell back to the labeled synthetic generator
    (``synthetic_fallback:*``) — or was simply unavailable — is **excluded
    from the score entirely** rather than blended in as if it were real:
    the composite is a renormalized average over only the signals that are
    actually trusted. If every financial AND every satellite signal ends
    up excluded, there is nothing legitimate left to weigh, and this
    returns ``MomentumLevel.INSUFFICIENT_DATA`` with no fabricated number —
    never a score computed partly or wholly from synthetic inputs
    presented as if it were real.
    """
    # None (never a fabricated number) whenever the underlying signal is
    # itself INSUFFICIENT_DATA -- see _signal_value() above.
    revenue = _signal_value(financial_signals["revenue_growth_signal"])
    rd = _signal_value(financial_signals["rd_investment_signal"])
    capex = _signal_value(financial_signals["capex_growth_signal"])
    # None (never a fabricated number) exactly when no verified facility
    # coordinates existed for this assessment, so vision inference was
    # never attempted at all -- see VisionFindings.status /
    # agents/orchestrator.py. The trust-gate below (satellite_trusted, from
    # data_sources) independently excludes these from the score in that
    # case; this guard only prevents a crash multiplying None by 100.
    construction = 100 * vision_findings.construction_expansion_signal if vision_findings.construction_expansion_signal is not None else None
    clearing = 100 * vision_findings.vegetation_clearing_signal if vision_findings.vegetation_clearing_signal is not None else None

    signal_values = {
        "revenue_growth": revenue,
        "rd_investment": rd,
        "capex_growth": capex,
        "construction_expansion": construction,
        "vegetation_clearing": clearing,
    }

    sources = data_sources or {}
    satellite_trusted = _is_trusted_source(sources.get("satellite_imagery_current")) and _is_trusted_source(
        sources.get("satellite_imagery_prior")
    )
    trusted = {
        "revenue_growth": _is_trusted_source(sources.get("revenue_growth_signal")),
        "rd_investment": _is_trusted_source(sources.get("rd_investment_signal")),
        "capex_growth": _is_trusted_source(sources.get("capex_growth_signal")),
        # Both satellite signals come from the same bi-temporal tile pair,
        # so they share one trust decision -- a pair that's half-live,
        # half-insufficient is not something the vision model can
        # meaningfully separate after the fact.
        "construction_expansion": satellite_trusted,
        "vegetation_clearing": satellite_trusted,
    }
    # Belt-and-suspenders: a signal whose value genuinely came back None
    # (INSUFFICIENT_DATA) is never trusted regardless of what the source
    # label says -- a missing value can never be legitimately averaged
    # into the score.
    for name in ("revenue_growth", "rd_investment", "capex_growth"):
        if signal_values[name] is None:
            trusted[name] = False

    # Reported honestly rather than silently dropped -- surfaces in
    # IntelligenceAssessmentResult so the Evidence/Data Quality views can
    # explain exactly why a signal is missing from the breakdown.
    excluded_signals = sorted(name for name, is_trusted in trusted.items() if not is_trusted)
    signal_breakdown = {
        name: value for name, value in signal_values.items() if trusted[name] and value is not None
    }

    financial_names = ("revenue_growth", "rd_investment", "capex_growth")
    satellite_names = ("construction_expansion", "vegetation_clearing")
    financial_kept = [signal_values[n] for n in financial_names if trusted[n]]
    satellite_kept = [signal_values[n] for n in satellite_names if trusted[n]]

    financial_composite = sum(financial_kept) / len(financial_kept) if financial_kept else None
    satellite_composite = sum(satellite_kept) / len(satellite_kept) if satellite_kept else None

    if financial_composite is None and satellite_composite is None:
        return MomentumAssessmentResult(
            momentum_score=0.0,
            momentum_level=MomentumLevel.INSUFFICIENT_DATA,
            signal_breakdown=signal_breakdown,  # empty -- nothing was trusted
            recommended_actions=[
                "No live financial or satellite data could be retrieved for this assessment -- a "
                "competitive-momentum score cannot be legitimately calculated. Retry once SEC EDGAR "
                "and/or NASA GIBS are reachable rather than relying on this result."
            ],
            excluded_signals=excluded_signals,
            financial_composite=None,
            satellite_composite=None,
            convergence_bonus=0.0,
        )

    # Convergence bonus: independent financial evidence (audited SEC
    # filings) and independent physical evidence (satellite imagery) both
    # pointing to expansion is a materially stronger signal than either
    # alone -- the same cross-validation logic real alternative-data
    # analysts apply before citing a signal in a research note, reducing
    # false positives from any single data source. Requires both composites
    # to exist (i.e. neither side was fully excluded) as well as both
    # clearing the threshold.
    convergence_bonus = 0.0
    if (
        financial_composite is not None
        and satellite_composite is not None
        and financial_composite >= CONVERGENCE_THRESHOLD
        and satellite_composite >= CONVERGENCE_THRESHOLD
    ):
        agreement = min(financial_composite, satellite_composite) / 100
        convergence_bonus = CONVERGENCE_BONUS_MAX * agreement

    if financial_composite is not None and satellite_composite is not None:
        base_score = FINANCIAL_WEIGHT * financial_composite + SATELLITE_WEIGHT * satellite_composite
    elif financial_composite is not None:
        # Satellite side fully excluded -- reweight to 100% financial
        # rather than silently treating the missing half as "0 change".
        base_score = financial_composite
    else:
        # Financial side fully excluded -- reweight to 100% satellite.
        base_score = satellite_composite

    score = max(0.0, min(100.0, base_score + convergence_bonus))
    level = classify_momentum_level(score)

    return MomentumAssessmentResult(
        momentum_score=score,
        momentum_level=level,
        signal_breakdown=signal_breakdown,
        recommended_actions=_recommend_actions(signal_breakdown, level),
        excluded_signals=excluded_signals,
        financial_composite=financial_composite,
        satellite_composite=satellite_composite,
        convergence_bonus=convergence_bonus,
    )


_ACTION_LIBRARY = {
    "revenue_growth": (
        "Model the revenue trajectory against sector peers to quantify any market-share shift."
    ),
    "rd_investment": (
        "Track R&D spend against patent filings and product-launch cadence to anticipate the next product cycle."
    ),
    "capex_growth": (
        "Cross-reference the capex trend with hiring data and supplier/contractor announcements to size the "
        "capacity build-out."
    ),
    "construction_expansion": (
        "Corroborate the satellite construction signal with local permitting filings and press coverage before "
        "citing it in a client deliverable."
    ),
    "vegetation_clearing": (
        "Flag the site for a follow-up imagery pass in 60-90 days to confirm the land-clearing converts into a "
        "completed structure."
    ),
}


def _recommend_actions(signal_breakdown: dict[str, float], level: MomentumLevel) -> list[str]:
    actions: list[str] = []
    ranked = sorted(signal_breakdown.items(), key=lambda kv: kv[1], reverse=True)
    top_signal, top_value = ranked[0]

    if top_value >= 40:
        actions.append(_ACTION_LIBRARY[top_signal])

    if level in (MomentumLevel.STRONG, MomentumLevel.AGGRESSIVE_EXPANSION):
        actions.append("Add this company/facility to the competitive intelligence watchlist for quarterly re-assessment.")
        actions.append("Brief the account or deal team ahead of the next earnings call or client engagement touchpoint.")
    if level is MomentumLevel.AGGRESSIVE_EXPANSION:
        actions.append(
            "Escalate to the engagement lead: convergent financial and satellite evidence warrants a dedicated "
            "competitor deep-dive."
        )
    if not actions:
        actions.append("No notable momentum signal; re-assess on the standard quarterly cadence.")
    return actions
