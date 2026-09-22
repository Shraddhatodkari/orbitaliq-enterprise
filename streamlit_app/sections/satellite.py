"""Satellite Intelligence — before/after NASA GIBS imagery, the diff
heatmap, and full provenance for one assessment, plus an optional live
re-fetch at a different lookback window. Every image byte and every
resolution/date/provenance field comes straight from the backend
(``data/imagery_provider.py::build_satellite_visual_package`` /
``GET /api/v1/satellite/change-pair``) — this page only decodes the
base64 PNG data URIs the backend already rendered.
"""
from __future__ import annotations

import base64

import streamlit as st

from streamlit_app.api_client import ApiError, OrbitalIQClient
from streamlit_app.components import assessment_picker, kpi_row, render_api_error, section_header, source_badge


def _decode_data_uri(data_uri: str | None) -> bytes | None:
    if not data_uri or "," not in data_uri:
        return None
    try:
        return base64.b64decode(data_uri.split(",", 1)[1])
    except Exception:
        return None


def _render_visual_package(visual: dict) -> None:
    kpi_row(
        [
            ("Current Observation", visual.get("current_observation_date") or "—", None),
            ("Prior Observation", visual.get("prior_observation_date") or "—", f"lookback {visual.get('lookback_days', '—')}d"),
            (
                "Resolution",
                f"{visual.get('resolution_m_per_pixel', 0):,.0f} m/px" if visual.get("resolution_m_per_pixel") else "—",
                "effective, after resize",
            ),
            ("Satellite Product", visual.get("satellite_product") or "—", None),
        ]
    )

    coords = visual.get("coordinates") or {}
    st.caption(
        f"Coordinates: {coords.get('latitude', '—')}, {coords.get('longitude', '—')} · "
        f"Source: [{visual.get('source_name', 'NASA GIBS')}]({visual.get('source_url', '')})"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Prior (before)**")
        img = _decode_data_uri(visual.get("prior_image_png"))
        if img:
            st.image(img, width="stretch")
        source_badge("nasa_gibs_live" if visual.get("prior_image_available") else "synthetic_fallback:satellite_tile")
    with c2:
        st.markdown("**Current (after)**")
        img = _decode_data_uri(visual.get("current_image_png"))
        if img:
            st.image(img, width="stretch")
        source_badge("nasa_gibs_live" if visual.get("current_image_available") else "synthetic_fallback:satellite_tile")
    with c3:
        st.markdown("**Change heatmap**")
        img = _decode_data_uri(visual.get("diff_image_png"))
        if img:
            st.image(img, width="stretch")
        st.caption("Red = larger per-pixel change")

    st.caption(visual.get("change_score_note", ""))


def render(client: OrbitalIQClient) -> None:
    section_header(
        "Satellite Intelligence",
        "Bi-temporal NASA GIBS change detection — before/after imagery, a per-pixel change heatmap, and full "
        "provenance, resolution, and date disclosure. Never confirmation of a specific business event, only "
        "large-scale land-cover change (see the note beneath each pair).",
    )

    assessment = assessment_picker(client, key="satellite_picker")
    if assessment is None:
        return

    st.markdown(f"### {assessment['company_name']} — {assessment['facility_name']}")

    has_coordinates = assessment.get("latitude") is not None and assessment.get("longitude") is not None
    visual = assessment.get("satellite_visual")
    if visual:
        st.markdown("**As captured for this assessment**")
        _render_visual_package(visual)
    elif not has_coordinates:
        st.warning(
            "INSUFFICIENT_DATA — verified location unavailable for this assessment (no verified facility "
            "coordinates; a financial-only assessment — see Company Intelligence for the financial data that "
            "IS available). Satellite intelligence was never attempted rather than estimated from a guessed "
            "location."
        )
    else:
        st.info(
            "This assessment has no attached satellite visual package (created while live data mode was "
            "disabled, or a satellite tile pair could not be built)."
        )

    st.divider()
    if not has_coordinates:
        st.caption(
            "Fetching fresh imagery requires verified facility coordinates, which this assessment does not "
            "have. Run a new assessment with confirmed coordinates (advanced fields) to use this tool."
        )
        return

    section_header(
        "Fetch fresh imagery for these coordinates",
        "Independent of the assessment above — calls the live change-pair endpoint directly so you can explore "
        "a different lookback window without creating a new assessment.",
    )
    lookback_days = st.slider("Lookback window (days)", min_value=7, max_value=730, value=180, step=7)
    if st.button("Fetch imagery", key="fetch_fresh_satellite"):
        try:
            resp = client.get_satellite_change_pair(
                latitude=assessment["latitude"], longitude=assessment["longitude"], lookback_days=lookback_days
            )
        except ApiError as err:
            render_api_error(err, context="Fetching satellite imagery")
            return

        if not resp.get("live_data_mode"):
            st.warning("Live data mode is disabled on this backend — no NASA GIBS request was made.")
            return
        if resp.get("visual") is None:
            st.warning(resp.get("unavailable_reason") or "No imagery available for this location/date pair.")
            return
        _render_visual_package(resp["visual"])
