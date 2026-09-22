"""Evidence & Audit + Data Quality — the full Claim → Source → Document →
Retrieved → Value → Calculation → Dashboard-signal audit trail
(``GET /api/v1/assessments/{id}/evidence``) and the honest source-health
report (``GET /api/v1/assessments/{id}/data-quality``) for one assessment.
Nothing here is computed locally; a value is either present with its real
source, or explicitly labeled insufficient/unavailable — never silently
filled in.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from streamlit_app.api_client import ApiError, OrbitalIQClient
from streamlit_app.components import assessment_picker, kpi_row, render_api_error, render_badge, section_header


def render(client: OrbitalIQClient) -> None:
    section_header(
        "Evidence & Audit / Data Quality",
        "The full audit trail behind one assessment's numbers, and an honest report of exactly which sources "
        "were live vs. fallback vs. unavailable.",
    )

    assessment = assessment_picker(client, key="evidence_quality_picker")
    if assessment is None:
        return

    st.caption(f"Assessment ID: `{assessment['id']}`")

    try:
        evidence_resp = client.get_evidence(assessment["id"])
        dq_resp = client.get_data_quality(assessment["id"])
    except ApiError as err:
        render_api_error(err, context="Loading evidence / data quality")
        return

    tab_evidence, tab_quality = st.tabs(["Evidence & Audit Trail", "Data Quality"])

    with tab_evidence:
        items = evidence_resp.get("evidence", [])
        if not items:
            st.info("No evidence items recorded for this assessment.")
        else:
            sec_items = [i for i in items if "sec" in (i.get("source_name") or "").lower() or "sec" in (i.get("source_url") or "").lower()]
            nasa_items = [i for i in items if "nasa" in (i.get("source_name") or "").lower() or "gibs" in (i.get("source_url") or "").lower()]
            kpi_row(
                [
                    ("Total Evidence Items", len(items), None),
                    ("SEC EDGAR Items", len(sec_items), None),
                    ("NASA GIBS Items", len(nasa_items), None),
                    ("Other Items", len(items) - len(sec_items) - len(nasa_items), None),
                ]
            )
            st.write("")
            df = pd.DataFrame(items)
            preferred_cols = [
                c
                for c in [
                    "claim", "source_name", "source_url", "document", "retrieved_at", "value", "calculation",
                    "dashboard_signal", "agent",
                ]
                if c in df.columns
            ]
            st.dataframe(df[preferred_cols] if preferred_cols else df, width="stretch", hide_index=True)
            with st.expander("Underlying signals referenced"):
                signals = sorted({i.get("dashboard_signal") for i in items if i.get("dashboard_signal")})
                for s in signals:
                    st.write(f"- {s}")
            with st.expander("Audit-grade identity (evidence_id / assessment_id / correlation_id / model)"):
                id_cols = [c for c in ["evidence_id", "assessment_id", "correlation_id", "agent", "model"] if c in df.columns]
                if id_cols:
                    st.dataframe(df[id_cols], width="stretch", hide_index=True)
                else:
                    st.caption("No audit-identity fields recorded for this assessment.")

    with tab_quality:
        dq = dq_resp.get("data_quality")
        if not dq:
            st.info("No data-quality summary recorded for this assessment.")
        else:
            kpi_row(
                [
                    (
                        "Completeness",
                        f"{dq.get('completeness_pct', 0):.0f}%",
                        "tracked financial/satellite signals only — see Public Evidence below",
                    ),
                    (
                        "Evidence Coverage",
                        f"{dq.get('evidence_coverage_pct', 0):.0f}%",
                        "live + disclosed fallback, tracked signals only",
                    ),
                    (
                        "Overall Rating",
                        dq.get("overall_rating", "—"),
                        "rates tracked signals only, not the full public-evidence picture",
                    ),
                    ("Live Signals", f"{dq.get('live_signals', 0)}/{dq.get('total_signals', 0)}", None),
                ]
            )
            render_badge("STRICT MODE" if dq.get("strict_mode") else "STANDARD MODE", "accent" if dq.get("strict_mode") else "neutral")
            st.write("")

            categories = dq.get("categories") or {}
            if categories:
                st.markdown("**Per-category rollup**")
                cat_cols = st.columns(len(categories))
                cat_labels = {"financial": "Financial", "satellite": "Satellite", "public_evidence": "Public Evidence"}
                for col, (cat_key, cat) in zip(cat_cols, categories.items()):
                    with col:
                        st.caption(cat_labels.get(cat_key, cat_key))
                        if cat.get("total_signals", 0) == 0:
                            st.metric(label="", value="N/A")
                            st.caption(cat.get("note") or "No signals defined for this category.")
                        else:
                            st.metric(label="", value=f"{cat.get('completeness_pct', 0):.0f}%")
                            st.caption(f"{cat.get('live_signals', 0)}/{cat.get('total_signals', 0)} live")
                st.write("")

            missing = dq.get("missing_fields") or []
            fallback = dq.get("fallback_fields") or []
            unavailable = dq.get("unavailable_fields") or []
            if missing:
                st.warning(f"**Insufficient-data warnings** — {len(missing)} field(s) not from a live source: {', '.join(missing)}")
            else:
                st.success(
                    "No missing metrics among the tracked financial/satellite signals — every one of them was live "
                    "for this assessment. (Public/external disclosure is a separate category — see below.)"
                )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Fallback disclosure**")
                if fallback:
                    for f in fallback:
                        st.write(f"- {f}")
                else:
                    st.caption("No fallback fields.")
            with c2:
                st.markdown("**Unavailable fields**")
                if unavailable:
                    for f in unavailable:
                        st.write(f"- {f}")
                else:
                    st.caption("No unavailable fields.")

            st.write("")
            st.markdown("**Freshness**")
            freshness = dq.get("freshness") or {}
            freshness_days = dq.get("freshness_days") or {}
            for label, date_key, days_key in (
                ("Most recent fiscal period", "most_recent_fiscal_period", None),
                ("Satellite current observation", "satellite_current_observation_date", "satellite_current_days_ago"),
                ("Satellite prior observation", "satellite_prior_observation_date", "satellite_prior_days_ago"),
                ("Assessed at", "assessed_at", "assessed_at_days_ago"),
            ):
                raw = freshness.get(date_key) or "—"
                days = freshness_days.get(days_key) if days_key else None
                suffix = f" ({days}d ago)" if days is not None else ""
                st.write(f"- {label}: {raw}{suffix}")

            with st.expander("Full source provenance map"):
                provenance = dq.get("provenance") or {}
                if provenance:
                    st.dataframe(
                        pd.DataFrame({"signal": list(provenance.keys()), "source": list(provenance.values())}),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.caption("No provenance map recorded.")
