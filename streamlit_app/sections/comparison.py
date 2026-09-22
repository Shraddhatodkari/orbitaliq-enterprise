"""Competitive Comparison — select 2–10 assessments and compare them
side by side. The ranking and every score shown comes straight from
``POST /api/v1/compare`` (a disclosed, deterministic sort by
momentum_score — never a subjective ranking); the financial/satellite
evidence panels underneath pull each assessment's own already-computed
record via ``GET /api/v1/assessments/{id}``, never recomputing anything.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

# See sections/agent_trace.py's identical guard for the rationale.
try:
    import plotly.graph_objects as go

    _PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when plotly truly isn't installed
    go = None  # type: ignore[assignment]
    _PLOTLY_AVAILABLE = False

from streamlit_app.api_client import ApiError, OrbitalIQClient
from streamlit_app.components import (
    SATELLITE_LIMITATION_CAVEAT,
    empty_state,
    fmt_expansion_signal_index,
    fmt_usd,
    render_api_error,
    section_header,
    source_badge,
)


def render(client: OrbitalIQClient) -> None:
    section_header(
        "Competitive Comparison",
        "Pick 2–10 assessments to compare side by side. Sorted by momentum_score descending — a disclosed, "
        "deterministic sort, never a subjective ranking.",
    )

    try:
        listing = client.list_assessments(limit=200)
    except ApiError as err:
        render_api_error(err, context="Loading the assessment list")
        return

    results = listing.get("results", [])
    if len(results) < 2:
        empty_state("Need at least 2 assessments to compare. Run more from **New Assessment**.")
        return

    results = sorted(results, key=lambda r: r.get("created_at") or "", reverse=True)
    options = {f"{r['ticker']} — {r['facility_name']} ({r['id'][:8]})": r["id"] for r in results}
    selected_labels = st.multiselect(
        "Select 2–10 assessments", list(options.keys()), default=list(options.keys())[: min(3, len(options))]
    )

    if len(selected_labels) < 2:
        st.info("Select at least 2 assessments.")
        return
    if len(selected_labels) > 10:
        st.warning("Select at most 10 — showing the first 10 selected.")
        selected_labels = selected_labels[:10]

    selected_ids = [options[label] for label in selected_labels]

    if st.button("Compare", type="primary"):
        try:
            compare_resp = client.compare_assessments(selected_ids)
        except ApiError as err:
            render_api_error(err, context="Comparing assessments")
            return

        results_sorted = compare_resp.get("results", [])
        missing_ids = compare_resp.get("missing_ids", [])
        if missing_ids:
            st.warning(f"{len(missing_ids)} selected assessment(s) could not be found and were skipped: {missing_ids}")

        st.caption(f"Sorted by: `{compare_resp.get('sorted_by')}`")

        summary_rows = []
        for r in results_sorted:
            dq = r.get("data_quality") or {}
            summary_rows.append(
                {
                    "Rank": len(summary_rows) + 1,
                    "Ticker": r["ticker"],
                    "Company": r["company_name"],
                    "Facility": r["facility_name"],
                    "Competitive Expansion Signal": r["momentum_level"],
                    "Composite Index": fmt_expansion_signal_index(r),
                    "Data Completeness": f"{dq.get('completeness_pct', 0):.0f}%" if dq else "—",
                    "Narrative Status": r.get("narrative_status", "—"),
                }
            )
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
        st.caption(SATELLITE_LIMITATION_CAVEAT)

        # Signal breakdown radar comparison.
        if not _PLOTLY_AVAILABLE:
            st.warning("Radar comparison chart unavailable — `plotly` is not installed (`pip install -r requirements.txt`).")
        else:
            fig = go.Figure()
            signal_labels = None
            for r in results_sorted:
                breakdown = r.get("signal_breakdown") or {}
                if not breakdown:
                    continue
                if signal_labels is None:
                    signal_labels = list(breakdown.keys())
                fig.add_trace(
                    go.Scatterpolar(
                        r=[breakdown.get(k, 0) for k in signal_labels],
                        theta=signal_labels,
                        fill="toself",
                        name=r["ticker"],
                    )
                )
            if signal_labels:
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    height=420,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                st.plotly_chart(fig, width="stretch")

        st.divider()
        section_header("Financial & satellite evidence detail", "Pulled per-assessment from the backend's own record — never recomputed.")
        for assessment_id in [r["id"] for r in results_sorted]:
            try:
                full = client.get_assessment(assessment_id)
            except ApiError as err:
                render_api_error(err, context=f"Loading detail for {assessment_id[:8]}")
                continue
            with st.expander(f"{full['ticker']} — {full['facility_name']} (signal: {full['momentum_level']}, {fmt_expansion_signal_index(full)})"):
                fc1, fc2 = st.columns(2)
                with fc1:
                    st.markdown("**Financial evidence**")
                    profile = full.get("financial_profile")
                    if profile and profile.get("metrics"):
                        m = profile["metrics"]
                        for key in ("revenue_growth", "rd_growth", "capex_growth"):
                            metric = m.get(key)
                            if not metric:
                                continue
                            st.write(
                                f"- **{metric['label']}**: {metric['status']} — "
                                f"{fmt_usd(metric.get('current_value')) if metric.get('current_value') is not None else 'N/A'}"
                                + (f" (YoY {metric['yoy_change_pct']*100:+.1f}%)" if metric.get("yoy_change_pct") is not None else "")
                            )
                    else:
                        st.caption("No real financial profile attached to this assessment.")
                with fc2:
                    st.markdown("**Satellite evidence**")
                    visual = full.get("satellite_visual")
                    if visual:
                        st.write(f"- Current: {visual.get('current_observation_date')}")
                        st.write(f"- Prior: {visual.get('prior_observation_date')}")
                        st.write(f"- Resolution: {visual.get('resolution_m_per_pixel', 0):,.0f} m/px")
                        source_badge("nasa_gibs_live" if visual.get("current_image_available") else "synthetic_fallback:satellite_tile")
                    else:
                        st.caption("No satellite visual attached to this assessment.")
                st.caption(f"Evidence items on record: {len(full.get('evidence') or [])} · Assessment ID: `{full['id']}`")
