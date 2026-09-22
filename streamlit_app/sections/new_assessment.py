"""New Assessment — the "type a company name" front door.

The user's primary input is just a Company Name. This page first calls
``POST /api/v1/companies/resolve`` to turn that free-text name into a
verified ticker/CIK (SEC EDGAR's own directory) and, when one can be
genuinely established, a verified facility location (the company's real
SEC-registered business address, geocoded via the US Census Bureau) —
never a fabricated ticker, company, or coordinate. Only once that
resolution has happened (or the user has filled in the advanced/manual
fields themselves) does this page submit to the existing
``POST /api/v1/assessments`` to actually run the real backend pipeline.
This page does not run any agent itself and does not compute anything —
it only submits the form and renders whatever the backend returns.
"""
from __future__ import annotations

import streamlit as st

from streamlit_app.api_client import ApiError, OrbitalIQClient
from streamlit_app.components import (
    SATELLITE_LIMITATION_CAVEAT,
    fmt_expansion_signal_index,
    is_insufficient_data,
    render_api_error,
    render_badge,
    section_header,
)

_AGENT_LABELS = {
    "ingestion_agent": "1. Ingestion Agent — SEC EDGAR + NASA GIBS retrieval",
    "vision_agent": "2. Vision Agent — satellite change-detection CNN",
    "intelligence_scoring_agent": "3. Intelligence Scoring Agent — deterministic Competitive Expansion Signal",
    "report_agent": "4. Report Agent — grounded narrative generation",
    "watchlist_agent": "5. Watchlist Agent — tier assignment + escalation",
}

_STATUS_BADGE = {
    "RESOLVED": "ok",
    "RESOLVED_NO_FACILITY": "warn",
    "AMBIGUOUS": "warn",
    "COMPANY_NOT_FOUND": "bad",
}


def _agent_trace_status(status: str) -> str:
    return {"ok": "ok", "error": "bad"}.get(status, "neutral")


def _init_default_state() -> None:
    """Seed every advanced-field widget's session_state key exactly once
    (on first render), before any widget is instantiated, so widgets never
    need a ``value=`` default competing with session_state — the standard
    Streamlit pattern for a widget whose value is later set
    programmatically (see ``_seed_advanced_fields``).
    """
    st.session_state.setdefault("adv_company_name", "")
    st.session_state.setdefault("adv_ticker", "")
    st.session_state.setdefault("adv_facility_name", "")
    st.session_state.setdefault("adv_has_coords", False)
    st.session_state.setdefault("adv_latitude_input", 0.0)
    st.session_state.setdefault("adv_longitude_input", 0.0)


def _seed_advanced_fields(resolution: dict | None, *, fallback_company_name: str = "") -> None:
    """Programmatically overwrite the advanced/override widgets' own
    session_state keys from a fresh resolution result (writing directly to
    each widget's ``key=`` — not a separate shadow variable — is what
    makes this actually take effect on the widget's next render; a
    ``value=`` kwarg on the widget itself is only honored the very first
    time that key is created). Only called right after a NEW resolution
    arrives, so a user's own subsequent manual edits to these fields on an
    unrelated rerun are never clobbered.
    """
    resolution = resolution or {}
    st.session_state["adv_company_name"] = resolution.get("company_name") or fallback_company_name
    st.session_state["adv_ticker"] = resolution.get("ticker") or ""
    st.session_state["adv_facility_name"] = resolution.get("facility_name") or ""
    lat, lon = resolution.get("latitude"), resolution.get("longitude")
    st.session_state["adv_has_coords"] = lat is not None and lon is not None
    st.session_state["adv_latitude_input"] = float(lat) if lat is not None else 0.0
    st.session_state["adv_longitude_input"] = float(lon) if lon is not None else 0.0


def render(client: OrbitalIQClient) -> None:
    section_header(
        "New Assessment",
        "Enter a company name — OrbitalIQ resolves the ticker, SEC CIK, and (when a real SEC-registered "
        "business address can be verified and geocoded) a facility location automatically. Then runs the "
        "real 5-agent pipeline (Ingestion → Vision → Scoring → Report → Watchlist) against live SEC EDGAR "
        "and NASA GIBS data when the backend has live mode enabled.",
    )

    _init_default_state()

    query = st.text_input(
        "Company Name*", placeholder="Tesla, Microsoft, Apple…", key="company_name_query",
        help="Any company name. OrbitalIQ only resolves companies that file with the SEC — a private or "
        "non-SEC-reporting company will honestly report COMPANY_NOT_FOUND rather than a guess.",
    )
    if st.button("Resolve Company", type="primary"):
        if not query or not query.strip():
            st.error("Enter a company name.")
        else:
            with st.spinner(f"Resolving '{query.strip()}' against SEC EDGAR…"):
                try:
                    resolution = client.resolve_company(query.strip())
                except ApiError as err:
                    st.session_state["resolution"] = None
                    render_api_error(err, context="Resolving company name")
                else:
                    st.session_state["resolution"] = resolution
                    _seed_advanced_fields(resolution, fallback_company_name=query.strip())

    resolution = st.session_state.get("resolution")
    if resolution:
        _render_resolution(client, resolution)

    st.divider()
    _render_run_form(client)

    _render_last_result()


def _render_resolution(client: OrbitalIQClient, resolution: dict) -> None:
    status = resolution.get("status", "—")
    render_badge(status, _STATUS_BADGE.get(status, "neutral"))

    if status == "COMPANY_NOT_FOUND":
        st.error(resolution.get("reason") or "No SEC-registered company matches this name.")
        st.caption(
            "This company may be private, non-SEC-reporting, or misspelled. You can still run a manual "
            "assessment using the advanced fields below if you already know the ticker/facility/coordinates."
        )
        return

    if status == "AMBIGUOUS":
        st.warning(resolution.get("reason") or "Multiple companies match this name — select the correct one.")
        candidates = resolution.get("candidates") or []
        options = {f"{c['ticker']} — {c['company_name']} (CIK {c['cik']})": c for c in candidates}
        if not options:
            return
        choice_label = st.selectbox("Select the correct company", list(options.keys()), key="ambiguous_choice")
        if st.button("Use this company"):
            chosen = options[choice_label]
            with st.spinner(f"Resolving {chosen['ticker']}…"):
                try:
                    resolved = client.resolve_company(chosen["ticker"])
                except ApiError as err:
                    render_api_error(err, context="Resolving selected company")
                    return
            st.session_state["resolution"] = resolved
            _seed_advanced_fields(resolved, fallback_company_name=chosen["company_name"])
            st.rerun()
        return

    # RESOLVED or RESOLVED_NO_FACILITY
    st.success(f"Resolved: **{resolution.get('company_name')}** ({resolution.get('ticker')}) — CIK {resolution.get('cik')}")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Ticker", resolution.get("ticker") or "—")
    with cols[1]:
        st.metric("Facility", resolution.get("facility_name") or "—")
    with cols[2]:
        if resolution.get("latitude") is not None and resolution.get("longitude") is not None:
            st.metric("Coordinates", f"{resolution['latitude']:.4f}, {resolution['longitude']:.4f}")
        else:
            st.metric("Coordinates", "INSUFFICIENT_DATA")

    if status == "RESOLVED_NO_FACILITY":
        st.warning(
            "INSUFFICIENT_DATA — verified location unavailable. "
            f"{resolution.get('facility_note') or ''} Financial intelligence can still run in full; satellite "
            "intelligence will be reported as insufficient data rather than an estimated location. Enter "
            "confirmed coordinates in the Advanced / Analyst Override section below if you have them from "
            "another source."
        )
    else:
        st.caption(resolution.get("facility_note") or "")


def _render_run_form(client: OrbitalIQClient) -> None:
    with st.form("new_assessment_form", clear_on_submit=False):
        st.markdown("**Advanced / Analyst Override — Optional**")
        st.caption("Pre-filled automatically from company resolution above. Not required for normal use.")
        c1, c2 = st.columns(2)
        with c1:
            company_name_override = st.text_input("Company Name (override)", key="adv_company_name")
            ticker = st.text_input("Ticker", max_chars=20, key="adv_ticker")
            facility_name = st.text_input("Facility Name", key="adv_facility_name")
            industry = st.text_input("Industry", value="general", key="adv_industry")
        with c2:
            has_coords = st.checkbox(
                "I have confirmed facility coordinates",
                key="adv_has_coords",
                help="Leave unchecked to run a financial-only assessment when no verified facility location "
                "exists — satellite intelligence will honestly report INSUFFICIENT_DATA rather than an "
                "estimated location.",
            )
            latitude = st.number_input(
                "Latitude", min_value=-90.0, max_value=90.0, format="%.4f",
                disabled=not has_coords, key="adv_latitude_input",
            )
            longitude = st.number_input(
                "Longitude", min_value=-180.0, max_value=180.0, format="%.4f",
                disabled=not has_coords, key="adv_longitude_input",
            )
            market_cap_usd = st.number_input(
                "Market Cap (USD, optional)", min_value=0.0, value=0.0, step=1_000_000.0, format="%.0f"
            )
        st.caption(
            "* Company Name is the only required input above. Ticker/Facility/Coordinates are auto-filled by "
            "Resolve Company when authoritative data permits, and can be overridden here for analyst review."
        )
        submitted = st.form_submit_button("Run Assessment", type="primary", width="stretch")

    if not submitted:
        return

    company_name = (company_name_override or "").strip()
    ticker = (ticker or "").strip()
    facility_name = (facility_name or "").strip()

    missing = [
        label for label, val in (("Company Name", company_name), ("Ticker", ticker), ("Facility Name", facility_name))
        if not val
    ]
    if missing:
        st.error(
            f"Please fill in: {', '.join(missing)}. Use \"Resolve Company\" above, or fill in the advanced "
            "fields manually."
        )
        return

    payload = {
        "ticker": ticker,
        "company_name": company_name,
        "facility_name": facility_name,
        "industry": (industry or "general").strip() or "general",
    }
    if has_coords:
        payload["latitude"] = latitude
        payload["longitude"] = longitude
    if market_cap_usd and market_cap_usd > 0:
        payload["market_cap_usd"] = market_cap_usd

    with st.status("Running the assessment pipeline…", expanded=True) as status:
        st.write("Submitting to the backend — this runs all 5 agents in one request, including live network calls.")
        try:
            result = client.create_assessment(payload)
        except ApiError as err:
            status.update(label="Assessment failed.", state="error")
            render_api_error(err, context="Running the assessment")
            return
        status.update(label="Assessment complete.", state="complete")

    st.session_state["last_assessment"] = result
    st.session_state["last_assessment_id"] = result["id"]
    if is_insufficient_data(result):
        st.warning(
            "Assessment ran, but a Competitive Expansion Signal could not be legitimately calculated — every "
            "underlying financial (SEC EDGAR) and satellite (NASA GIBS) signal for this facility came back "
            "unavailable or fell back to a labeled synthetic placeholder. See Data Quality below rather than "
            "treating this as a real signal."
        )
    else:
        st.success(
            f"Assessment complete — Competitive Expansion Signal: {result['momentum_level']} "
            f"({fmt_expansion_signal_index(result)})."
        )
        if result.get("latitude") is None or result.get("longitude") is None:
            st.info(
                "Financial-only assessment — INSUFFICIENT_DATA — verified location unavailable, so satellite "
                "intelligence did not run."
            )
        st.caption(SATELLITE_LIMITATION_CAVEAT)
    _render_agent_trace(result)
    _render_summary(result)


def _render_last_result() -> None:
    result = st.session_state.get("last_assessment")
    if not result:
        return
    st.divider()
    st.caption("Most recent result this session:")
    _render_agent_trace(result)
    _render_summary(result)


def _render_agent_trace(result: dict) -> None:
    st.markdown("**Pipeline progress by agent** (as reported by the backend's own trace)")
    trace = result.get("agent_trace") or []
    for entry in trace:
        agent = entry.get("agent", "")
        label = _AGENT_LABELS.get(agent, agent)
        cols = st.columns([5, 1, 2])
        with cols[0]:
            st.markdown(f"**{label}**")
            st.caption(entry.get("summary", ""))
        with cols[1]:
            render_badge(entry.get("status", "").upper() or "—", _agent_trace_status(entry.get("status", "")))
        with cols[2]:
            st.caption(f"{entry.get('duration_ms', 0):.1f} ms")


def _render_summary(result: dict) -> None:
    with st.expander("Narrative report", expanded=True):
        render_badge(result.get("narrative_status", "—"), "ok" if result.get("narrative_status") == "AI_GENERATED_GROUNDED" else "neutral")
        st.write("")
        st.write(result.get("narrative_report", ""))
    with st.expander("Recommended actions"):
        for action in result.get("recommended_actions", []):
            st.markdown(f"- {action}")
    st.caption(
        f"Assessment ID: `{result['id']}` — open **Company Intelligence**, **Satellite Intelligence**, "
        "**Evidence & Audit**, or **Agent Trace** in the sidebar to inspect this assessment in full."
    )
