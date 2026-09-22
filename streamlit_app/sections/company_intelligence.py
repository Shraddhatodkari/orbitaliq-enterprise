"""Company Intelligence — the per-assessment financial + investment +
satellite-change view, read entirely from one already-persisted
``AssessmentResponse``. No ratio, growth rate, or score is recalculated
here; every number is exactly what the backend's SEC-EDGAR-only
``FinancialProfile`` or deterministic scoring engine already produced.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

# See sections/agent_trace.py's identical guard for the rationale.
try:
    import plotly.express as px

    _PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when plotly truly isn't installed
    px = None  # type: ignore[assignment]
    _PLOTLY_AVAILABLE = False

from streamlit_app.api_client import OrbitalIQClient
from streamlit_app.components import (
    SATELLITE_LIMITATION_CAVEAT,
    assessment_picker,
    fmt_expansion_signal_index,
    fmt_pct,
    fmt_usd,
    kpi_row,
    render_badge,
    section_header,
    source_badge,
    status_pill_for_metric_status,
)

_GROWTH_METRICS = ["revenue_growth", "rd_growth", "capex_growth", "operating_income_growth", "free_cash_flow_trend"]
_INVESTMENT_METRICS = ["capex_to_revenue", "rd_to_revenue", "asset_growth", "ppe_growth"]
_PROFITABILITY_METRICS = ["gross_margin", "operating_margin", "net_margin", "roa", "roe"]
_BALANCE_SHEET_METRICS = ["cash", "total_debt", "net_debt", "current_ratio", "debt_to_equity"]


def _metric_row(metrics: dict, key: str) -> dict | None:
    return metrics.get(key)


def _render_metric_table(metrics: dict, keys: list[str]) -> None:
    rows = []
    for k in keys:
        m = metrics.get(k)
        if not m:
            continue
        is_pct = m.get("unit") == "percent"
        is_ratio = m.get("unit") == "ratio"
        if is_pct:
            current = fmt_pct(m.get("current_value"))
            prior = fmt_pct(m.get("prior_value"))
        elif is_ratio:
            current = f"{m['current_value']:.2f}x" if m.get("current_value") is not None else "—"
            prior = f"{m['prior_value']:.2f}x" if m.get("prior_value") is not None else "—"
        else:
            current = fmt_usd(m.get("current_value"))
            prior = fmt_usd(m.get("prior_value"))
        rows.append(
            {
                "Metric": m.get("label", k),
                "Status": m.get("status", "—"),
                "Current": current,
                "Prior": prior,
                "YoY Change": fmt_pct(m.get("yoy_change_pct")),
                "Fiscal Period": m.get("fiscal_period") or "—",
                "XBRL Concept": m.get("xbrl_concept") or "—",
                "Note": m.get("note") or "",
            }
        )
    if not rows:
        st.info("No metrics in this category.")
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


def render(client: OrbitalIQClient) -> None:
    section_header(
        "Company Intelligence",
        "Real financial metrics (SEC EDGAR XBRL), investment intensity, satellite change signals, and the "
        "deterministic Competitive Expansion Signal for one assessment.",
    )

    assessment = assessment_picker(client, key="company_intel_picker")
    if assessment is None:
        return

    st.markdown(f"### {assessment['company_name']} ({assessment['ticker']}) — {assessment['facility_name']}")

    kpi_row(
        [
            ("Competitive Expansion Signal", assessment.get("momentum_level", "—"), fmt_expansion_signal_index(assessment)),
            (
                "Narrative Status",
                assessment.get("narrative_status", "—").replace("_", " ").title(),
                None,
            ),
            ("Watchlist Tier", assessment.get("watchlist_tier", "NONE"), assessment.get("watchlist_channel") or "not escalated"),
            ("Created", (assessment.get("created_at") or "—")[:19].replace("T", " "), None),
        ]
    )
    st.caption(SATELLITE_LIMITATION_CAVEAT)

    st.write("")
    section_header("Deterministic signal breakdown (0–100 index, feeds the Competitive Expansion Signal)")
    breakdown = assessment.get("signal_breakdown") or {}
    if not breakdown:
        st.info("No signal breakdown recorded for this assessment.")
    elif not _PLOTLY_AVAILABLE:
        st.warning("Chart unavailable — `plotly` is not installed (`pip install -r requirements.txt`).")
        st.dataframe(
            pd.DataFrame({"signal": list(breakdown.keys()), "index": [round(v, 1) for v in breakdown.values()]}),
            width="stretch", hide_index=True,
        )
    else:
        df_breakdown = pd.DataFrame(
            {"signal": list(breakdown.keys()), "index": [round(v, 1) for v in breakdown.values()]}
        )
        fig = px.bar(df_breakdown, x="index", y="signal", orientation="h", range_x=[0, 100])
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    st.write("")
    section_header("Why this signal? (component breakdown, evidence coverage, drill-down)")
    score_breakdown = assessment.get("score_breakdown")
    if not score_breakdown:
        st.info("No score breakdown recorded for this assessment (assessments created before this feature shipped).")
    else:
        weights = score_breakdown.get("weights") or {}
        kpi_row(
            [
                (
                    "Financial Contribution",
                    "—" if score_breakdown.get("financial_contribution") is None
                    else f"{score_breakdown['financial_contribution']:.1f}",
                    f"weight {weights.get('financial_weight', '—')}",
                ),
                (
                    "Satellite Contribution",
                    "—" if score_breakdown.get("satellite_contribution") is None
                    else f"{score_breakdown['satellite_contribution']:.1f}",
                    f"weight {weights.get('satellite_weight', '—')}",
                ),
                (
                    "Convergence Bonus",
                    f"+{score_breakdown.get('convergence_bonus', 0):.1f}",
                    f"max {weights.get('convergence_bonus_max', '—')}",
                ),
                (
                    "Evidence Coverage",
                    f"{score_breakdown.get('data_confidence_pct', 0):.0f}%",
                    "share of THIS assessment's tracked financial/satellite signals with valid "
                    "LIVE/CACHED_REAL/DERIVED evidence — not a measure of overall public-evidence completeness",
                ),
            ]
        )
        if score_breakdown.get("contradictory_signals"):
            st.warning(
                "⚠️ Contradictory signals: financial and satellite evidence point in opposing directions "
                f"({score_breakdown.get('convergence_outcome')}) — {score_breakdown.get('convergence_explanation', '')}"
            )
        with st.expander("Per-signal drill-down — why each signal was included or excluded"):
            explanations = score_breakdown.get("signal_explanations") or []
            if explanations:
                df_expl = pd.DataFrame(
                    [
                        {
                            "Signal": e.get("signal"),
                            "Trusted": "✓" if e.get("trusted") else "✗ excluded",
                            "Value (0-100)": "—" if e.get("value_0_100") is None else f"{e['value_0_100']:.1f}",
                            "Source": e.get("source") or "—",
                            "Reason": e.get("reason"),
                            "What it measures": e.get("description"),
                        }
                        for e in explanations
                    ]
                )
                st.dataframe(df_expl, width="stretch", hide_index=True)
            else:
                st.info("No per-signal explanations recorded.")

    st.write("")
    section_header("Source provenance for this assessment")
    sources = assessment.get("data_sources") or {}
    cols = st.columns(max(len(sources), 1))
    for col, (k, v) in zip(cols, sources.items()):
        with col:
            st.caption(k)
            source_badge(v)

    st.write("")
    profile = assessment.get("financial_profile")
    if not profile:
        st.warning(
            "No real-data financial profile is attached to this assessment — it was likely created while "
            "`ORBITALIQ_LIVE_DATA_MODE` was disabled, or the ticker could not be resolved against SEC EDGAR's "
            "company directory. The signal breakdown above (deterministic scoring engine) can still use its own "
            "labeled synthetic fallback where disclosed, but this Real Financial Intelligence view has nothing "
            "to show without a real SEC filing."
        )
        return

    st.caption(
        f"SEC EDGAR CIK: `{profile.get('cik') or 'not resolved'}` · retrieved {profile.get('retrieved_at', '—')}"
        + (f" · [filing source]({profile['filing_source_url']})" if profile.get("filing_source_url") else "")
    )

    metrics = profile.get("metrics") or {}
    tabs = st.tabs(["Growth", "Investment Intensity", "Profitability", "Balance Sheet"])
    with tabs[0]:
        _render_metric_table(metrics, _GROWTH_METRICS)
    with tabs[1]:
        _render_metric_table(metrics, _INVESTMENT_METRICS)
    with tabs[2]:
        _render_metric_table(metrics, _PROFITABILITY_METRICS)
    with tabs[3]:
        _render_metric_table(metrics, _BALANCE_SHEET_METRICS)

    st.write("")
    section_header("Satellite change signals (from this assessment's vision findings)")
    vision = assessment.get("vision_findings") or {}
    if vision and vision.get("status") == "INSUFFICIENT_DATA":
        st.warning(
            "INSUFFICIENT_DATA — verified location unavailable. Vision inference was never attempted for "
            "this assessment (never estimated from a guessed location) — see Satellite Intelligence for detail."
        )
    elif vision:
        kpi_row(
            [
                ("Overall Change Score", f"{vision.get('overall_change_score') or 0:.2f}", "0–1 scale"),
                ("Construction Expansion", f"{vision.get('construction_expansion_signal') or 0:.2f}", None),
                ("Vegetation Clearing", f"{vision.get('vegetation_clearing_signal') or 0:.2f}", None),
                ("Device Used", vision.get("device_used", "—"), "fine-tuned" if vision.get("model_finetuned") else "seeded init"),
            ]
        )
    else:
        st.info("No vision findings recorded for this assessment.")

    st.write("")
    section_header("Convergence & evidence")
    convergence = assessment.get("convergence")
    if convergence:
        c1, c2, c3 = st.columns(3)
        for col, label, key in (
            (c1, "Financial", "financial_level"),
            (c2, "Physical (Satellite)", "physical_level"),
            (c3, "External / Public Disclosure", "external_level"),
        ):
            with col:
                st.caption(label)
                level = convergence.get(key, "—")
                render_badge(level, {"HIGH": "ok", "MODERATE": "warn", "LOW": "bad", "INSUFFICIENT": "neutral"}.get(level, "neutral"))
        st.write("")
        outcome = convergence.get("outcome", "—")
        render_badge(
            outcome,
            {
                "SIGNAL_CONVERGENCE": "ok",
                "PARTIAL_SIGNAL_CONVERGENCE": "warn",
                "SIGNAL_CONFLICT": "bad",
                "INSUFFICIENT_EVIDENCE": "neutral",
            }.get(outcome, "neutral"),
        )
        st.write("")
        st.write(convergence.get("explanation", ""))
    else:
        st.info("No convergence classification recorded for this assessment.")

    with st.expander(f"Evidence trail ({len(assessment.get('evidence') or [])} items)"):
        evidence = assessment.get("evidence") or []
        if not evidence:
            st.info("No evidence items recorded.")
        else:
            df_ev = pd.DataFrame(evidence)
            st.dataframe(df_ev, width="stretch", hide_index=True)
