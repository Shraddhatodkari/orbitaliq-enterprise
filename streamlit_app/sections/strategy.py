"""Executive Strategy View — a concise, analyst-style briefing composed
ENTIRELY from fields the backend already computed for one assessment:
the (grounding-critic-checked) narrative, the convergence classification,
the data-quality gap list, and the watchlist tier. This page only
templates already-labeled values into hedged prose — it never asks the
LLM (or anything else) to produce a new number, and it never displays a
narrative that failed grounding without saying so.
"""
from __future__ import annotations

import streamlit as st

from streamlit_app.api_client import OrbitalIQClient
from streamlit_app.components import assessment_picker, render_badge, section_header


def render(client: OrbitalIQClient) -> None:
    section_header(
        "Executive Strategy View",
        "A templated brief over already-computed fields — key evidence, expansion indicators, contradictory "
        "signals, data gaps, and watch items. No new numbers are generated on this page.",
    )

    assessment = assessment_picker(client, key="strategy_picker")
    if assessment is None:
        return

    st.markdown(f"### {assessment['company_name']} ({assessment['ticker']}) — {assessment['facility_name']}")

    narrative_status = assessment.get("narrative_status", "—")
    status_color = {
        "AI_GENERATED_GROUNDED": "ok",
        "AI_GENERATED_BLOCKED_REPLACED_WITH_DETERMINISTIC": "warn",
        "DETERMINISTIC_GROUNDED_TEMPLATE": "neutral",
    }.get(narrative_status, "neutral")
    render_badge(narrative_status.replace("_", " "), status_color)
    if narrative_status == "AI_GENERATED_BLOCKED_REPLACED_WITH_DETERMINISTIC":
        st.caption(
            "The LLM-generated narrative cited a number the grounding critic could not verify against this "
            "assessment's own computed values, so it was discarded and replaced with the deterministic template "
            "below — never displayed unsupported."
        )
    st.write("")
    st.write(assessment.get("narrative_report", ""))

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Expansion indicators**")
        breakdown = assessment.get("signal_breakdown") or {}
        strong_signals = sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)[:3]
        if strong_signals:
            for name, value in strong_signals:
                st.write(f"- {name.replace('_', ' ').title()}: {value:.1f} / 100")
        else:
            st.caption("No signal breakdown available.")

        st.markdown("**Watch items**")
        tier = assessment.get("watchlist_tier", "NONE")
        if tier == "NONE":
            # tier "NONE" covers BOTH a genuine STABLE result and
            # INSUFFICIENT_DATA (no legitimate score could be calculated at
            # all -- see agents/watchlist_agent.py's TIER_BY_MOMENTUM_LEVEL,
            # which deliberately never escalates either case). Read the
            # real momentum_level rather than assuming STABLE, so an
            # INSUFFICIENT_DATA assessment is never mislabeled as a genuine
            # stable result.
            actual_level = assessment.get("momentum_level", "—")
            st.caption(f"Not currently on the watchlist (momentum level: {actual_level}).")
        else:
            st.write(f"- Escalated to **{tier}** — {assessment.get('watchlist_channel') or 'no channel configured'}")
            st.write(f"- Webhook dispatch: {assessment.get('watchlist_webhook_status', '—')}")

    with col2:
        st.markdown("**Contradictory signals**")
        convergence = assessment.get("convergence")
        if convergence and convergence.get("outcome") == "SIGNAL_CONFLICT":
            st.warning(convergence.get("explanation", "Financial and satellite evidence point in different directions."))
        elif convergence:
            st.caption(f"No conflict detected — {convergence.get('outcome', '—').replace('_', ' ').lower()}.")
        else:
            st.caption("No convergence classification available.")

        st.markdown("**Data gaps (tracked financial/satellite signals)**")
        dq = assessment.get("data_quality")
        if dq:
            gap_fields = (dq.get("fallback_fields") or []) + (dq.get("unavailable_fields") or [])
            if gap_fields:
                for f in gap_fields:
                    st.write(f"- {f.replace('_', ' ')}")
            else:
                st.caption("No gaps among the tracked financial/satellite signals — every one of them was live.")
            st.caption(
                f"Evidence coverage (tracked signals only): {dq.get('completeness_pct', 0):.0f}% — "
                "excludes external/public disclosure, which has no data source wired into the pipeline yet "
                "and is reported separately as INSUFFICIENT, not counted here as a gap."
            )
        else:
            st.caption("No data-quality summary available.")

    st.divider()
    st.markdown("**Recommended next steps**")
    for action in assessment.get("recommended_actions", []):
        st.markdown(f"- {action}")
