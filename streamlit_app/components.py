"""Reusable, presentation-only UI building blocks shared by every section.

Nothing in this file computes a business value — it only formats and lays
out values it is handed. Where a value might be missing, it renders the
honest label the backend already assigned (``INSUFFICIENT_DATA``, ``Not
Available``, ``None``) rather than substituting a placeholder number.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from streamlit_app.api_client import ApiError

# --- Status color language (used consistently across every section) -------
# ok (green): a genuine live source / passed check.
# warn (amber): a disclosed fallback / partial signal / caution.
# bad (red): insufficient / unavailable / failed.
# neutral (grey): not applicable / no opinion.
_STATUS_COLORS = {
    "ok": "#0f9d58",
    "warn": "#b8860b",
    "bad": "#c0392b",
    "neutral": "#6b7280",
    "accent": "#2454a6",
}


def inject_base_style() -> None:
    st.markdown(
        """
        <style>
        .oiq-kpi-card {
            border: 1px solid rgba(49, 51, 63, 0.15);
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            background: rgba(36, 84, 166, 0.04);
            height: 100%;
        }
        .oiq-kpi-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #6b7280;
            margin-bottom: 0.15rem;
        }
        .oiq-kpi-value {
            font-size: 1.6rem;
            font-weight: 700;
            line-height: 1.15;
        }
        .oiq-kpi-sub {
            font-size: 0.8rem;
            color: #6b7280;
            margin-top: 0.15rem;
        }
        .oiq-badge {
            display: inline-block;
            padding: 0.12rem 0.55rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            color: white;
            white-space: nowrap;
        }
        .oiq-section-note {
            color: #6b7280;
            font-size: 0.85rem;
            margin-top: -0.4rem;
            margin-bottom: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge(label: str, status: str = "neutral") -> str:
    color = _STATUS_COLORS.get(status, _STATUS_COLORS["neutral"])
    return f'<span class="oiq-badge" style="background:{color}">{label}</span>'

def render_badge(label: str, status: str = "neutral") -> None:
    st.markdown(badge(label, status), unsafe_allow_html=True)


def source_status(source: str | None) -> str:
    """Classify a backend-disclosed ``source`` / ``data_sources`` value
    into the shared ok/warn/bad/neutral language — the same rule the
    existing JS dashboard already uses (see dashboard/app.js::sourceBadge),
    reproduced here so both UIs read provenance identically.

    Mirrors the canonical 5-state contract in
    ``core/data_quality_state.py::classify_state`` (this file deliberately
    stays decoupled from the ``orbitaliq`` backend package — see the module
    docstring — so the rule is reproduced rather than imported):

    - ``*_live`` (LIVE) and ``derived:*`` (DERIVED) are both genuine, real
      values -> "ok".
    - ``cached_real:*`` is a genuine, previously-observed real value, but
      explicitly flagged stale -> "warn".
    - ``synthetic_fallback:*`` / ``offline_demo_mode`` are disclosed,
      non-production demo/dev-only labels, never a live-mode failure ->
      "warn" (a caution, not a hard failure).
    - ``error:*``, ``NOT_AVAILABLE``, and any ``insufficient_data*`` label
      are all a genuine absence of a real value -> "bad".
    """
    if not source:
        return "neutral"
    s = str(source)
    if s == "offline_demo_mode" or s.startswith("synthetic_fallback"):
        return "warn"
    if s.startswith("cached_real"):
        return "warn"
    if s.endswith("_live") or s.startswith("derived"):
        return "ok"
    if s.startswith("error") or s == "NOT_AVAILABLE" or "insufficient" in s:
        return "bad"
    return "neutral"


def source_badge(source: str | None) -> None:
    if not source:
        render_badge("NOT AVAILABLE", "neutral")
        return
    render_badge(str(source).upper(), source_status(source))


def kpi_card(label: str, value: Any, sub: str | None = None) -> None:
    sub_html = f'<div class="oiq-kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="oiq-kpi-card">
            <div class="oiq-kpi-label">{label}</div>
            <div class="oiq-kpi-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple[str, Any, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        with col:
            kpi_card(label, value, sub)


def fmt_number(value: Any, *, decimals: int = 1, suffix: str = "") -> str:
    """Render a possibly-``None`` numeric value honestly. Never coerces a
    missing value into 0 or any other placeholder.
    """
    if value is None:
        return "—"
    try:
        return f"{float(value):,.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def fmt_usd(value: Any, *, compact: bool = True) -> str:
    if value is None:
        return "Not Disclosed"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not compact:
        return f"${v:,.0f}"
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e12:
        return f"{sign}${v / 1e12:,.2f}T"
    if v >= 1e9:
        return f"{sign}${v / 1e9:,.2f}B"
    if v >= 1e6:
        return f"{sign}${v / 1e6:,.2f}M"
    if v >= 1e3:
        return f"{sign}${v / 1e3:,.1f}K"
    return f"{sign}${v:,.0f}"


def fmt_pct(value: Any, *, already_fraction: bool = True, decimals: int = 1) -> str:
    """``value`` is a fraction (0.12 -> "12.0%") unless
    ``already_fraction=False``, in which case it is treated as already
    being on a 0-100 scale.
    """
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if already_fraction:
        v *= 100
    sign = "+" if v > 0 else ""
    return f"{sign}{v:,.{decimals}f}%"


def status_pill_for_metric_status(status: str | None) -> tuple[str, str]:
    """(label, status-color-key) for a FinancialMetric's own
    ``status`` field ("AVAILABLE" | "INSUFFICIENT_DATA").
    """
    if status == "AVAILABLE":
        return "AVAILABLE", "ok"
    return "INSUFFICIENT DATA", "bad"


def momentum_level_status(level: str | None) -> str:
    return {
        "STABLE": "neutral",
        "EMERGING": "warn",
        "STRONG": "accent",
        "AGGRESSIVE_EXPANSION": "ok",
        "INSUFFICIENT_DATA": "bad",
    }.get(level or "", "neutral")


def is_insufficient_data(record: dict) -> bool:
    """True when the backend could not legitimately calculate a
    competitive-momentum score for this assessment (every underlying
    financial/satellite signal was a live-mode fallback or unavailable —
    see ``core/intelligence_engine.py``). The record's numeric
    ``momentum_score`` is a fixed ``0.0`` placeholder in that case, never a
    real score, so no view should ever display it as one.
    """
    return bool(record) and record.get("momentum_level") == "INSUFFICIENT_DATA"


def fmt_momentum_score(record: dict) -> str:
    if is_insufficient_data(record):
        return "Insufficient Data"
    score = record.get("momentum_score")
    return f"{score:.1f} / 100" if isinstance(score, (int, float)) else "—"


# --- Output philosophy (task #73): the qualitative Competitive Expansion
# Signal level is the headline every view leads with; the 0-100 composite
# index is real, disclosed supporting detail -- never hidden, but never the
# primary framing, because a precise-looking number invites a false sense
# of confidence a qualitative signal level doesn't. ---------------------

SATELLITE_LIMITATION_CAVEAT = (
    "Satellite evidence is directional only — large-scale land/construction change detected from public "
    "VIIRS/MODIS imagery — and does not confirm the specific use, ownership, or purpose of any change."
)


def fmt_expansion_signal(record: dict) -> str:
    """The qualitative headline value: the Competitive Expansion Signal
    level itself (e.g. "STRONG", "INSUFFICIENT_DATA")."""
    return record.get("momentum_level") or "—"


def fmt_expansion_signal_index(record: dict) -> str:
    """The supporting-detail composite index for the headline above --
    the same real number ``fmt_momentum_score`` renders, just always
    phrased as secondary detail rather than the lead value."""
    if is_insufficient_data(record):
        return "no composite index — insufficient live data"
    score = record.get("momentum_score")
    return f"composite index {score:.1f}/100" if isinstance(score, (int, float)) else "—"


def watchlist_tier_status(tier: str | None) -> str:
    return {
        "NONE": "neutral",
        "WATCH": "warn",
        "REVIEW": "accent",
        "HIGH_SIGNAL": "bad",
    }.get(tier or "", "neutral")


def render_api_error(err: ApiError, *, context: str) -> None:
    """The single, shared "show the real error clearly" surface every
    section uses — see requirement: "Show errors clearly instead of
    displaying a generic failure." Always shows the pipeline stage, the
    real backend message, the source involved, and a request id when the
    backend provided one, instead of a bare "something went wrong."
    """
    st.error(f"**{context} failed.**")
    with st.container(border=True):
        for line in err.as_display_lines():
            st.markdown(line)
    if err.stage == "connection":
        st.info(
            "The Streamlit dashboard could not reach the FastAPI backend. Start it with "
            "`uvicorn orbitaliq.main:app --host 0.0.0.0 --port 8000` in a separate terminal, "
            "then retry."
        )


def empty_state(message: str) -> None:
    st.info(message)


def section_header(title: str, note: str | None = None) -> None:
    st.subheader(title)
    if note:
        st.markdown(f'<div class="oiq-section-note">{note}</div>', unsafe_allow_html=True)


def assessment_picker(client: Any, *, key: str) -> dict | None:
    """A shared "pick which assessment to look at" control used by every
    per-assessment section (Company Intelligence, Satellite Intelligence,
    Agent Trace, Evidence & Audit, Executive Strategy View). Defaults to
    the most recently created assessment in this browser session (set by
    New Assessment) when one exists, otherwise the most recent one on
    record. Returns the full ``AssessmentResponse`` dict, or ``None`` if
    there is nothing to pick from.
    """
    try:
        listing = client.list_assessments(limit=200)
    except ApiError as err:
        render_api_error(err, context="Loading the assessment list")
        return None

    results = listing.get("results", [])
    if not results:
        empty_state("No assessments yet. Run one from **New Assessment** in the sidebar first.")
        return None

    results = sorted(results, key=lambda r: r.get("created_at") or "", reverse=True)
    options = {f"{r['ticker']} — {r['facility_name']} ({r['id'][:8]})": r for r in results}
    labels = list(options.keys())

    default_index = 0
    last_id = st.session_state.get("last_assessment_id")
    if last_id:
        for i, r in enumerate(results):
            if r["id"] == last_id:
                default_index = i
                break

    choice = st.selectbox("Assessment", labels, index=default_index, key=key)
    return options[choice]
