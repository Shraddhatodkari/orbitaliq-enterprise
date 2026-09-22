"""Agent Trace — the backend's own per-agent execution record for one
assessment (``agents/orchestrator.py``'s ``AgentTraceEntry`` list): status,
duration, and a summary that already encodes each agent's key
inputs/outputs (e.g. how many signals were live, the momentum score it
produced, which narrative source was used). This page renders that trace
verbatim — it does not run or re-time anything itself.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from streamlit_app.api_client import OrbitalIQClient
from streamlit_app.components import assessment_picker, render_badge, section_header

# Plotly is an optional dependency for this one page (the duration bar
# chart) -- everything else on this page (the trace table, per-agent
# expanders, total wall time) works without it. A missing/broken plotly
# install must degrade this page to a chart-less table, never crash the
# whole Streamlit app (see README > Known limitations and
# requirements.txt, which does pin plotly>=5.24,<6 -- this lazy import is
# a defense-in-depth measure for a stale/partial venv, not a substitute
# for installing requirements.txt correctly).
try:
    import plotly.express as px

    _PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when plotly truly isn't installed
    px = None  # type: ignore[assignment]
    _PLOTLY_AVAILABLE = False

_AGENT_LABELS = {
    "ingestion_agent": "Ingestion Agent",
    "vision_agent": "Vision Agent",
    "intelligence_scoring_agent": "Scoring Agent",
    "report_agent": "Report Agent",
    "watchlist_agent": "Watchlist Agent",
}


def render(client: OrbitalIQClient) -> None:
    section_header(
        "Agent Trace",
        "Per-agent execution status, duration, and summary (inputs/outputs and any error), exactly as "
        "recorded by the orchestrator for this assessment.",
    )

    assessment = assessment_picker(client, key="agent_trace_picker")
    if assessment is None:
        return

    trace = assessment.get("agent_trace") or []
    if not trace:
        st.info("No agent trace recorded for this assessment.")
        return

    df = pd.DataFrame(
        [
            {
                "Agent": _AGENT_LABELS.get(t.get("agent"), t.get("agent")),
                "Status": t.get("status"),
                "Duration (ms)": t.get("duration_ms"),
            }
            for t in trace
        ]
    )
    if _PLOTLY_AVAILABLE:
        fig = px.bar(
            df, x="Duration (ms)", y="Agent", orientation="h", color="Status",
            color_discrete_map={"ok": "#0f9d58", "error": "#c0392b"},
        )
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning(
            "Duration chart unavailable — the `plotly` package is not installed in this environment "
            "(`pip install -r requirements.txt` resolves this). Showing the trace table instead."
        )
        st.dataframe(df, width="stretch", hide_index=True)

    for t in trace:
        agent = t.get("agent", "")
        status = t.get("status", "")
        with st.expander(f"{_AGENT_LABELS.get(agent, agent)} — {status.upper()} ({t.get('duration_ms', 0):.1f} ms)", expanded=(status == "error")):
            render_badge(status.upper(), "ok" if status == "ok" else "bad")
            st.write("")
            st.markdown("**Summary (inputs/outputs as reported by this agent):**")
            st.write(t.get("summary", "—"))

    total_ms = sum(t.get("duration_ms", 0) for t in trace)
    st.caption(f"Total pipeline wall time reported: {total_ms:.1f} ms")
