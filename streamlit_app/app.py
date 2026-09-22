"""OrbitalIQ Streamlit analyst dashboard — entry point.

Run with:
    streamlit run streamlit_app/app.py --server.port 8501

This file only wires up page config, sidebar navigation, and a backend
connectivity check, then dispatches to one of the ``sections/*`` modules.
No business logic lives here — see ``streamlit_app/api_client.py`` for the
only place this app talks to the network, and each ``sections/*.py`` for
how a page renders what the backend returns.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run streamlit_app/app.py` to be launched from the repo
# root without an editable install — mirrors how tests/conftest.py adds
# `src/` to sys.path for the same reason.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from streamlit_app.api_client import ApiError, get_client
from streamlit_app.components import inject_base_style, render_api_error
from streamlit_app.config import APP_TITLE, PAGE_ICON, get_api_base_url
from streamlit_app.sections import (
    agent_trace,
    comparison,
    company_intelligence,
    evidence_quality,
    executive,
    new_assessment,
    satellite,
    strategy,
    watchlist,
)

st.set_page_config(page_title=APP_TITLE, page_icon=PAGE_ICON, layout="wide", initial_sidebar_state="expanded")
inject_base_style()

_SECTIONS = {
    "Executive Dashboard": executive,
    "New Assessment": new_assessment,
    "Company Intelligence": company_intelligence,
    "Satellite Intelligence": satellite,
    "Competitive Comparison": comparison,
    "Watchlist": watchlist,
    "Executive Strategy View": strategy,
    "Agent Trace": agent_trace,
    "Evidence & Audit / Data Quality": evidence_quality,
}


def main() -> None:
    client = get_client()

    with st.sidebar:
        st.markdown(f"## {PAGE_ICON} OrbitalIQ Enterprise")
        st.caption("Competitive Expansion Intelligence Platform — Analyst Workspace")
        choice = st.radio("Navigate", list(_SECTIONS.keys()), label_visibility="collapsed")

        st.divider()
        st.caption(f"Backend: `{get_api_base_url()}`")
        try:
            health = client.health()
            st.success("Backend connected")
            st.caption(
                f"live_data_mode={health.get('live_data_mode')} · "
                f"offline_mode={health.get('offline_mode')} · "
                f"vision_device={health.get('vision_device')}"
            )
        except ApiError as err:
            st.error("Backend unreachable")
            st.caption(err.message)

    st.title(choice)
    try:
        client.health()
    except ApiError as err:
        render_api_error(err, context="Connecting to the OrbitalIQ backend")
        st.stop()

    _SECTIONS[choice].render(client)


if __name__ == "__main__":
    main()
