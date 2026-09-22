"""Executive Dashboard — a fleet-level rollup over every assessment
already persisted by the backend. Every number here is either a straight
count/average over already-computed ``AssessmentResponse`` fields, or a
plain aggregation of the backend's own ``data_sources`` / ``narrative_status``
labels — nothing is scored or inferred here.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

# See agent_trace.py's identical guard for the rationale: plotly is
# optional per-page (charts only) so a missing/broken install degrades
# this page's two charts to a table/empty-state, never crashes the app.
try:
    import plotly.express as px

    _PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when plotly truly isn't installed
    px = None  # type: ignore[assignment]
    _PLOTLY_AVAILABLE = False

from streamlit_app.api_client import ApiError, OrbitalIQClient
from streamlit_app.components import (
    empty_state,
    fmt_expansion_signal_index,
    is_insufficient_data,
    kpi_row,
    momentum_level_status,
    render_api_error,
    render_badge,
    section_header,
)


def render(client: OrbitalIQClient) -> None:
    section_header(
        "Executive Dashboard",
        "Fleet-wide rollup across every assessment run so far — every figure below is a "
        "direct read or aggregation of already-computed backend values.",
    )

    try:
        assessments_resp = client.list_assessments(limit=200)
        watchlist_resp = client.list_watchlist(tier="NONE", limit=500)
    except ApiError as err:
        render_api_error(err, context="Loading the executive dashboard")
        return

    results = assessments_resp.get("results", [])
    if not results:
        empty_state(
            "No assessments yet. Run one from **New Assessment** in the sidebar to populate this dashboard."
        )
        return

    total_assessments = assessments_resp.get("total_matching", len(results))
    facilities_covered = {(r["ticker"], r["facility_name"]) for r in results}
    # A record with no legitimately-calculable score (momentum_level ==
    # "INSUFFICIENT_DATA") carries a fixed 0.0 placeholder — averaging it
    # in would silently drag the fleet average down with a number that was
    # never real. Excluded from the average and the histogram below;
    # counted separately instead.
    scored_results = [r for r in results if not is_insufficient_data(r)]
    insufficient_count = len(results) - len(scored_results)
    avg_momentum = (sum(r["momentum_score"] for r in scored_results) / len(scored_results)) if scored_results else None
    watchlisted = sum(1 for e in watchlist_resp.get("results", []) if e.get("tier") != "NONE")

    kpi_row(
        [
            ("Total Assessments", f"{total_assessments:,}", None),
            ("Companies / Facilities Covered", f"{len(facilities_covered):,}", None),
            (
                "Avg. Composite Index",
                f"{avg_momentum:,.1f} / 100" if avg_momentum is not None else "—",
                f"{insufficient_count} assessment(s) excluded — insufficient data" if insufficient_count else None,
            ),
            ("Currently On Watchlist", f"{watchlisted:,}", "WATCH + REVIEW + HIGH_SIGNAL"),
        ]
    )
    if insufficient_count:
        st.warning(
            f"{insufficient_count} of {len(results)} assessment(s) could not be legitimately scored "
            "(insufficient live data) and are excluded from every average/chart below — see each one's own "
            "Data Quality tab for details."
        )

    st.write("")

    # --- Financial / satellite signal health, aggregated from each
    # assessment's own `data_sources` provenance map (never re-derived).
    financial_keys = {"revenue_growth_signal", "rd_investment_signal", "capex_growth_signal"}
    satellite_keys = {"satellite_imagery_current", "satellite_imagery_prior"}
    fin_live = fin_total = sat_live = sat_total = 0
    completeness_values: list[float] = []
    grounded_count = 0
    for r in results:
        sources = r.get("data_sources") or {}
        for k, v in sources.items():
            is_live = str(v).endswith("_live")
            if k in financial_keys:
                fin_total += 1
                fin_live += int(is_live)
            elif k in satellite_keys:
                sat_total += 1
                sat_live += int(is_live)
        dq = r.get("data_quality") or {}
        if dq.get("completeness_pct") is not None:
            completeness_values.append(dq["completeness_pct"])
        if r.get("narrative_status") == "AI_GENERATED_GROUNDED":
            grounded_count += 1

    fin_pct = (fin_live / fin_total * 100) if fin_total else None
    sat_pct = (sat_live / sat_total * 100) if sat_total else None
    avg_completeness = (sum(completeness_values) / len(completeness_values)) if completeness_values else None
    grounded_pct = (grounded_count / len(results) * 100) if results else None

    kpi_row(
        [
            (
                "Financial Signals Live",
                f"{fin_pct:,.0f}%" if fin_pct is not None else "—",
                f"{fin_live}/{fin_total} SEC EDGAR calls succeeded" if fin_total else "no financial signals recorded",
            ),
            (
                "Satellite Signals Live",
                f"{sat_pct:,.0f}%" if sat_pct is not None else "—",
                f"{sat_live}/{sat_total} NASA GIBS tiles succeeded" if sat_total else "no satellite signals recorded",
            ),
            (
                "Avg. Data Completeness",
                f"{avg_completeness:,.0f}%" if avg_completeness is not None else "—",
                "Data Quality Center completeness_pct, averaged",
            ),
            (
                "AI-Grounded Narratives",
                f"{grounded_pct:,.0f}%" if grounded_pct is not None else "—",
                "passed the grounding critic vs. deterministic template",
            ),
        ]
    )

    st.write("")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Composite index distribution**")
        if not scored_results:
            empty_state("No legitimately-scored assessments yet.")
        elif not _PLOTLY_AVAILABLE:
            st.warning("Chart unavailable — `plotly` is not installed (`pip install -r requirements.txt`).")
            st.dataframe(
                pd.DataFrame({"momentum_score": [r["momentum_score"] for r in scored_results]}),
                width="stretch", hide_index=True,
            )
        else:
            df_scores = pd.DataFrame({"momentum_score": [r["momentum_score"] for r in scored_results]})
            fig = px.histogram(df_scores, x="momentum_score", nbins=20, range_x=[0, 100])
            fig.update_layout(
                height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Composite index", yaxis_title="Assessments"
            )
            st.plotly_chart(fig, width="stretch")

    with col_b:
        st.markdown("**Competitive Expansion Signal breakdown**")
        level_counts = Counter(r["momentum_level"] for r in results)
        df_levels = pd.DataFrame({"level": list(level_counts.keys()), "count": list(level_counts.values())})
        if not _PLOTLY_AVAILABLE:
            st.warning("Chart unavailable — `plotly` is not installed (`pip install -r requirements.txt`).")
            st.dataframe(df_levels, width="stretch", hide_index=True)
        else:
            fig2 = px.pie(df_levels, names="level", values="count", hole=0.55)
            fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, width="stretch")

    st.write("")
    section_header("Recent assessments")
    rows = []
    for r in sorted(results, key=lambda x: x.get("created_at") or "", reverse=True)[:25]:
        rows.append(
            {
                "Ticker": r["ticker"],
                "Company": r["company_name"],
                "Facility": r["facility_name"],
                "Competitive Expansion Signal": r["momentum_level"],
                "Composite Index": fmt_expansion_signal_index(r),
                "Watchlist Tier": r.get("watchlist_tier", "NONE"),
                "Narrative Status": r.get("narrative_status", "—"),
                "Created": r.get("created_at", "—"),
                "ID": r["id"],
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)

    with st.expander("Competitive Expansion Signal legend"):
        for level in ("STABLE", "EMERGING", "STRONG", "AGGRESSIVE_EXPANSION", "INSUFFICIENT_DATA"):
            render_badge(level, momentum_level_status(level))
            st.write("")
