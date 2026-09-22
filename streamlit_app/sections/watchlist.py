"""Watchlist — the deterministic-rules-only escalation list
(``agents/watchlist_agent.py``: STABLE→NONE, EMERGING→WATCH,
STRONG→REVIEW, AGGRESSIVE_EXPANSION→HIGH_SIGNAL), read from
``GET /api/v1/watchlist``. Tier assignment and webhook-dispatch status are
never computed here — only filtered and displayed.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from streamlit_app.api_client import ApiError, OrbitalIQClient
from streamlit_app.components import (
    empty_state,
    fmt_expansion_signal_index,
    render_api_error,
    render_badge,
    section_header,
    watchlist_tier_status,
)

_TIER_OPTIONS = ["(current watchlist only)", "NONE", "WATCH", "REVIEW", "HIGH_SIGNAL"]


def render(client: OrbitalIQClient) -> None:
    section_header(
        "Watchlist",
        "Deterministic tier assignment from the backend's rules engine — WATCH / REVIEW / HIGH_SIGNAL entries "
        "are companies whose most recent assessment crossed an EMERGING/STRONG/AGGRESSIVE_EXPANSION momentum "
        "threshold. Filter to NONE to see every company ever assessed, including de-escalated ones.",
    )

    tier_choice = st.radio("Filter", _TIER_OPTIONS, horizontal=True)
    tier_param = None if tier_choice == "(current watchlist only)" else tier_choice

    try:
        resp = client.list_watchlist(tier=tier_param, limit=500)
    except ApiError as err:
        render_api_error(err, context="Loading the watchlist")
        return

    entries = resp.get("results", [])
    if not entries:
        empty_state("Nothing to show for this filter yet.")
        return

    st.caption(f"{resp.get('total', len(entries))} entries")

    tier_counts = {t: sum(1 for e in entries if e.get("tier") == t) for t in ("HIGH_SIGNAL", "REVIEW", "WATCH", "NONE")}
    cols = st.columns(4)
    for col, tier in zip(cols, ("HIGH_SIGNAL", "REVIEW", "WATCH", "NONE")):
        with col:
            render_badge(f"{tier}: {tier_counts[tier]}", watchlist_tier_status(tier))

    st.write("")
    rows = []
    for e in entries:
        rows.append(
            {
                "Tier": e.get("tier"),
                "Ticker": e.get("ticker"),
                "Company": e.get("company_name"),
                "Facility": e.get("facility_name"),
                "Competitive Expansion Signal": e.get("momentum_level"),
                "Composite Index": fmt_expansion_signal_index(e),
                "Channel": e.get("channel") or "—",
                "Webhook Status": e.get("webhook_status"),
                "Reason": e.get("reason"),
                "First Flagged": (e.get("first_flagged_at") or "—")[:19].replace("T", " "),
                "Updated": (e.get("updated_at") or "—")[:19].replace("T", " "),
                "Latest Assessment ID": e.get("latest_assessment_id"),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.write("")
    section_header("Evidence for a watchlist entry")
    labels = {f"{e['ticker']} — {e['facility_name']} ({e['tier']})": e["latest_assessment_id"] for e in entries}
    choice = st.selectbox("Pick an entry", list(labels.keys()))
    if st.button("Show evidence for this entry"):
        assessment_id = labels[choice]
        try:
            evidence_resp = client.get_evidence(assessment_id)
        except ApiError as err:
            render_api_error(err, context="Loading evidence")
            return
        items = evidence_resp.get("evidence", [])
        if not items:
            st.info("No evidence items recorded for this assessment.")
        else:
            st.dataframe(pd.DataFrame(items), width="stretch", hide_index=True)
