"""Data Quality Center — an honest, per-assessment report on exactly how
much of the underlying evidence is real, live, and current, so a strict
enterprise mode can say plainly "this metric cannot be calculated" instead
of quietly filling a gap.

This module classifies every entry in an assessment's ``data_sources`` map
into one of three buckets and reports completeness/freshness/provenance
from that classification — it does not fetch anything itself. It also
rolls that same classification up per evidence category (Financial /
Satellite / Public Evidence) and into freshness-in-days and an overall
qualitative rating, so an executive can read "Financial: 100% complete,
Satellite: 50% complete, Public Evidence: not available" at a glance
instead of parsing a flat per-signal list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from orbitaliq.core.convergence import classify_level

NOT_AVAILABLE = "NOT_AVAILABLE"

# Which data_sources keys belong to which evidence category, matching the
# same three groups core/convergence.py's SignalConvergence already
# classifies (financial_level/physical_level/external_level) — this
# module's per-category rollup is deliberately aligned to that existing
# taxonomy rather than inventing a new one.
_FINANCIAL_SIGNAL_KEYS = ("revenue_growth_signal", "rd_investment_signal", "capex_growth_signal")
_SATELLITE_SIGNAL_KEYS = ("satellite_imagery_current", "satellite_imagery_prior")
# No public-disclosure data source is currently wired into this pipeline
# (see core/convergence.py's own note: external_level is always
# INSUFFICIENT). This category is still reported -- honestly, as empty --
# rather than omitted, so the Data Quality Center's shape is stable and a
# viewer can see plainly that this evidence group simply isn't collected
# yet, rather than wondering whether it was silently dropped.
_PUBLIC_EVIDENCE_SIGNAL_KEYS: tuple[str, ...] = ()

_CATEGORY_KEYS = {
    "financial": _FINANCIAL_SIGNAL_KEYS,
    "satellite": _SATELLITE_SIGNAL_KEYS,
    "public_evidence": _PUBLIC_EVIDENCE_SIGNAL_KEYS,
}


def _bucket(source: str) -> str:
    """Classifies a per-signal provenance label into one of three buckets.

    Deliberately kept as three distinct buckets rather than collapsing
    "fallback" and "unavailable": ``offline_demo_mode`` /
    ``synthetic_fallback:*`` are the disclosed, test/demo-only synthetic
    path (see ``data/sec_edgar_client.py``'s ``synthetic_financial_signals``
    docstring), while ``insufficient_data`` / ``NOT_AVAILABLE`` / an
    ``error:*`` label are a genuine live-mode production failure that
    returned no value at all (never a fabricated one -- see
    ``_insufficient_financial_signal``). Conflating the two would make it
    impossible to tell "this assessment ran in offline demo mode" apart
    from "this assessment tried and failed to reach SEC EDGAR / NASA GIBS
    live," which is exactly the distinction the Data Quality Center exists
    to preserve. ``cached_real:*`` and ``derived:*`` (see
    core/data_quality_state.py's 5-state contract) count as "live" here --
    both are genuine, trustworthy values, just not a fresh direct fetch;
    see the freshness-in-days fields below for how stale a CACHED_REAL
    value is.
    """
    if source == NOT_AVAILABLE or str(source).startswith("insufficient_data") or str(source).startswith("error"):
        return "unavailable"
    if str(source).startswith("synthetic_fallback") or source == "offline_demo_mode":
        return "fallback"
    if str(source).endswith("_live") or str(source).startswith("cached_real") or str(source).startswith("derived"):
        return "live"
    return "unavailable"


def _category_summary(data_sources: dict[str, str], keys: tuple[str, ...]) -> dict:
    """The same live/fallback/unavailable rollup as the top-level summary,
    scoped to one evidence category's signal keys.
    """
    present = {k: v for k, v in data_sources.items() if k in keys}
    total = len(keys)  # the category's *defined* size, not just what's present
    live_fields = [k for k, v in present.items() if _bucket(v) == "live"]
    fallback_fields = [k for k, v in present.items() if _bucket(v) == "fallback"]
    unavailable_fields = sorted(set(keys) - set(live_fields) - set(fallback_fields))
    completeness_pct = round(100.0 * len(live_fields) / total, 1) if total else 0.0
    return {
        "total_signals": total,
        "live_signals": len(live_fields),
        "fallback_signals": len(fallback_fields),
        "unavailable_signals": len(unavailable_fields),
        "completeness_pct": completeness_pct,
        "fields": sorted(keys),
        "note": (
            None if total else
            "No data source for this evidence category is currently wired into the pipeline — reported "
            "honestly as empty rather than omitted. See core/convergence.py's external_level, which is "
            "always INSUFFICIENT for the same reason."
        ),
    }


def _days_ago(iso_date: str | None, *, reference: datetime | None = None) -> int | None:
    """Whole days between an ISO date/datetime string and ``reference``
    (defaults to now, UTC). Returns ``None`` for anything that isn't a
    parseable date/datetime — e.g. a fiscal period like "FY2023" — rather
    than raising or guessing.
    """
    if not iso_date:
        return None
    ref = reference or datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(iso_date)
    except ValueError:
        try:
            parsed = datetime.combine(date.fromisoformat(iso_date), datetime.min.time(), tzinfo=timezone.utc)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (ref - parsed).days


@dataclass
class DataQualitySummary:
    strict_mode: bool
    total_signals: int
    live_signals: int
    fallback_signals: int
    unavailable_signals: int
    completeness_pct: float
    source_count: int
    missing_fields: list[str] = field(default_factory=list)
    fallback_fields: list[str] = field(default_factory=list)
    unavailable_fields: list[str] = field(default_factory=list)
    freshness: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    # --- Data Quality Center elevation (additive) ---
    # Per-category (Financial / Satellite / Public Evidence) rollup, each
    # shaped exactly like the top-level counts above (see
    # _category_summary) — lets a viewer see at a glance which evidence
    # group is weak, rather than only a flat per-signal list.
    categories: dict = field(default_factory=dict)
    # Freshness expressed as "days ago" (None where not computable, e.g. a
    # fiscal period isn't a calendar date) rather than only raw date
    # strings a reader has to do the subtraction on themselves.
    freshness_days: dict = field(default_factory=dict)
    # How much of this assessment has SOME grounded number at all (live,
    # cached-real, derived, OR a disclosed fallback) vs a blank gap --
    # distinct from completeness_pct, which counts only live/derived/
    # cached-real. A value can be "covered" (something is there, even if
    # only a disclosed demo fallback) without being "complete" (a genuine
    # live value).
    evidence_coverage_pct: float = 0.0
    # HIGH / MODERATE / LOW / INSUFFICIENT -- reuses
    # core/convergence.py's own classify_level() and thresholds (66/33),
    # so this rollup uses the exact same scale as the Convergence view
    # rather than inventing a second one.
    overall_rating: str = "INSUFFICIENT"

    def as_dict(self) -> dict:
        return {
            "strict_mode": self.strict_mode,
            "total_signals": self.total_signals,
            "live_signals": self.live_signals,
            "fallback_signals": self.fallback_signals,
            "unavailable_signals": self.unavailable_signals,
            "completeness_pct": self.completeness_pct,
            "source_count": self.source_count,
            "missing_fields": self.missing_fields,
            "fallback_fields": self.fallback_fields,
            "unavailable_fields": self.unavailable_fields,
            "freshness": self.freshness,
            "provenance": self.provenance,
            "categories": self.categories,
            "freshness_days": self.freshness_days,
            "evidence_coverage_pct": self.evidence_coverage_pct,
            "overall_rating": self.overall_rating,
        }


def build_data_quality_summary(
    *,
    data_sources: dict[str, str],
    strict_mode: bool,
    fiscal_period: str | None = None,
    satellite_current_date: str | None = None,
    satellite_prior_date: str | None = None,
    assessed_at: str | None = None,
) -> DataQualitySummary:
    total = len(data_sources)
    live_fields = [k for k, v in data_sources.items() if _bucket(v) == "live"]
    fallback_fields = [k for k, v in data_sources.items() if _bucket(v) == "fallback"]
    unavailable_fields = [k for k, v in data_sources.items() if _bucket(v) == "unavailable"]

    completeness_pct = round(100.0 * len(live_fields) / total, 1) if total else 0.0
    evidence_coverage_pct = round(100.0 * (len(live_fields) + len(fallback_fields)) / total, 1) if total else 0.0
    # "Source count" is the number of distinct authoritative systems
    # actually contributing live data to this assessment (e.g.
    # sec_edgar_live, nasa_gibs_live) — not the number of individual
    # signals, since one system typically backs several signals.
    distinct_sources = {
        data_sources[k].rsplit("_live", 1)[0].split(":", 1)[0] for k in live_fields
    }

    reference = datetime.now(timezone.utc)
    freshness_days = {
        "satellite_current_days_ago": _days_ago(satellite_current_date, reference=reference),
        "satellite_prior_days_ago": _days_ago(satellite_prior_date, reference=reference),
        "assessed_at_days_ago": _days_ago(assessed_at, reference=reference),
    }

    categories = {name: _category_summary(data_sources, keys) for name, keys in _CATEGORY_KEYS.items()}
    overall_rating = classify_level(completeness_pct, available=total > 0)

    return DataQualitySummary(
        strict_mode=strict_mode,
        total_signals=total,
        live_signals=len(live_fields),
        fallback_signals=len(fallback_fields),
        unavailable_signals=len(unavailable_fields),
        completeness_pct=completeness_pct,
        source_count=len(distinct_sources),
        missing_fields=sorted(fallback_fields + unavailable_fields),
        fallback_fields=sorted(fallback_fields),
        unavailable_fields=sorted(unavailable_fields),
        freshness={
            "most_recent_fiscal_period": fiscal_period,
            "satellite_current_observation_date": satellite_current_date,
            "satellite_prior_observation_date": satellite_prior_date,
            "assessed_at": assessed_at,
        },
        provenance=dict(data_sources),
        categories=categories,
        freshness_days=freshness_days,
        evidence_coverage_pct=evidence_coverage_pct,
        overall_rating=overall_rating,
    )
