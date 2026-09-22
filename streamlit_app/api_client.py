"""Thin HTTP client over the existing FastAPI backend.

This is the ONLY module in ``streamlit_app`` that talks to the network.
Every function here does exactly one thing: call one existing backend
endpoint and return its JSON body (or raise :class:`ApiError`). No
financial ratios, momentum scores, convergence classifications, or
narrative text are computed here — they are all already computed and
disclosed by the backend; this module just fetches them.

Error handling: every non-2xx response is translated into a single
:class:`ApiError` carrying the HTTP status, the pipeline **stage** that
failed (when the backend's structured error body provides one — see
``api/routes_assessment.py::_error_detail``), a human-readable message,
the data **source** involved (when known), and a **request_id** for
correlating with the backend's server-side logs. A connection failure
(backend not running, wrong port, timeout) is its own distinct, clearly
labeled error rather than being confused with a backend-side failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from streamlit_app.config import REQUEST_TIMEOUT_SECONDS, get_api_base_url


@dataclass
class ApiError(Exception):
    """A single, normalized shape for every way a backend call can fail.

    ``stage`` is one of the backend's own diagnostic stages when known
    (``pipeline_execution``, ``persistence``, ``response_serialization``),
    ``connection`` when the backend could not be reached at all, or
    ``http_error`` / ``validation_error`` for everything else (e.g. a 404,
    or a 422 the backend didn't wrap in the structured shape).
    """

    status_code: int | None
    stage: str
    message: str
    source: str | None = None
    ticker: str | None = None
    request_id: str | None = None
    raw_detail: Any = field(default=None, repr=False)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message

    def as_display_lines(self) -> list[str]:
        """Ordered (label, value) lines for rendering in ``st.error`` /
        an expander — never guesses at anything not already on this
        object.
        """
        lines = [f"**Stage:** {self.stage}"]
        if self.status_code is not None:
            lines.append(f"**HTTP status:** {self.status_code}")
        lines.append(f"**Message:** {self.message}")
        if self.source:
            lines.append(f"**Source:** {self.source}")
        if self.ticker:
            lines.append(f"**Ticker:** {self.ticker}")
        if self.request_id:
            lines.append(f"**Request ID:** `{self.request_id}`")
        return lines


def _parse_detail(status_code: int, detail: Any) -> ApiError:
    """Normalize any shape of FastAPI's ``detail`` field (see
    ``api/routes_assessment.py::_error_detail`` for the structured shape
    this backend now returns for assessment-creation failures, and plain
    strings/lists for everything else FastAPI produces automatically).
    """
    if isinstance(detail, dict) and ("stage" in detail or "message" in detail):
        return ApiError(
            status_code=status_code,
            stage=str(detail.get("stage") or "http_error"),
            message=str(detail.get("message") or "(no message provided)"),
            source=detail.get("source"),
            ticker=detail.get("ticker"),
            request_id=detail.get("request_id"),
            raw_detail=detail,
        )
    if isinstance(detail, list):
        # FastAPI/Pydantic's automatic 422 validation-error shape.
        parts = []
        for item in detail:
            if isinstance(item, dict):
                loc = ".".join(str(p) for p in item.get("loc", []) if p != "body")
                msg = item.get("msg", str(item))
                parts.append(f"{loc}: {msg}" if loc else msg)
            else:
                parts.append(str(item))
        return ApiError(
            status_code=status_code,
            stage="validation_error",
            message="; ".join(parts) or "Request validation failed.",
            raw_detail=detail,
        )
    if detail is None:
        return ApiError(status_code=status_code, stage="http_error", message="(no error body returned)")
    return ApiError(status_code=status_code, stage="http_error", message=str(detail), raw_detail=detail)


class OrbitalIQClient:
    """A small wrapper around :class:`httpx.Client`. One instance is
    cached per Streamlit session (see ``get_client`` below) so connection
    pooling actually helps across reruns within one browser session.
    """

    def __init__(self, base_url: str | None = None, timeout: float = REQUEST_TIMEOUT_SECONDS) -> None:
        self._client = httpx.Client(base_url=base_url or get_api_base_url(), timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise ApiError(
                status_code=None,
                stage="connection",
                message=(
                    f"Could not reach the OrbitalIQ backend at {self._client.base_url}. "
                    "Is the FastAPI server running (uvicorn orbitaliq.main:app)? "
                    f"Underlying error: {exc}"
                ),
                source="fastapi_backend",
            ) from exc
        except httpx.TimeoutException as exc:
            raise ApiError(
                status_code=None,
                stage="connection",
                message=(
                    f"The backend at {self._client.base_url} did not respond within "
                    f"{self._client.timeout.read}s. A live assessment can legitimately take a "
                    "while (real SEC EDGAR + NASA GIBS network calls) — if this keeps happening, "
                    "check the backend's own terminal for a stuck request."
                ),
                source="fastapi_backend",
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiError(status_code=None, stage="connection", message=str(exc), source="fastapi_backend") from exc

        if response.status_code >= 400:
            body: Any = None
            try:
                body = response.json()
            except Exception:
                body = None
            detail = body.get("detail") if isinstance(body, dict) else None
            raise _parse_detail(response.status_code, detail if detail is not None else response.text)

        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    # --- Health -----------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/health")

    # --- Assessments --------------------------------------------------

    def create_assessment(self, payload: dict) -> dict:
        return self._request("POST", "/api/v1/assessments", json=payload)

    def resolve_company(self, query: str) -> dict:
        """The "type a company name" front door — resolves a free-text
        query into a verified ticker/CIK/facility (or an honest
        AMBIGUOUS/COMPANY_NOT_FOUND result) via
        ``POST /api/v1/companies/resolve``. Never itself creates an
        assessment; see ``sections/new_assessment.py``.
        """
        return self._request("POST", "/api/v1/companies/resolve", json={"query": query})

    def get_assessment(self, assessment_id: str) -> dict:
        return self._request("GET", f"/api/v1/assessments/{assessment_id}")

    def list_assessments(
        self, *, min_momentum_score: float | None = None, limit: int = 100, offset: int = 0
    ) -> dict:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if min_momentum_score is not None:
            params["min_momentum_score"] = min_momentum_score
        return self._request("GET", "/api/v1/assessments", params=params)

    # --- Real financial / satellite intelligence -----------------------

    def get_financial_profile(self, ticker: str) -> dict:
        return self._request("GET", f"/api/v1/financials/{ticker}")

    def get_historical_financials(self, ticker: str, *, max_years: int = 6) -> dict:
        return self._request("GET", f"/api/v1/historical/{ticker}", params={"max_years": max_years})

    def get_satellite_change_pair(self, *, latitude: float, longitude: float, lookback_days: int = 180) -> dict:
        return self._request(
            "GET",
            "/api/v1/satellite/change-pair",
            params={"latitude": latitude, "longitude": longitude, "lookback_days": lookback_days},
        )

    # --- Per-assessment audit surfaces ---------------------------------

    def get_evidence(self, assessment_id: str) -> dict:
        return self._request("GET", f"/api/v1/assessments/{assessment_id}/evidence")

    def get_convergence(self, assessment_id: str) -> dict:
        return self._request("GET", f"/api/v1/assessments/{assessment_id}/convergence")

    def get_data_quality(self, assessment_id: str) -> dict:
        return self._request("GET", f"/api/v1/assessments/{assessment_id}/data-quality")

    # --- Comparison & watchlist -----------------------------------------

    def compare_assessments(self, assessment_ids: list[str]) -> dict:
        return self._request("POST", "/api/v1/compare", json={"assessment_ids": assessment_ids})

    def list_watchlist(self, *, tier: str | None = None, limit: int = 200, offset: int = 0) -> dict:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if tier is not None:
            params["tier"] = tier
        return self._request("GET", "/api/v1/watchlist", params=params)


def get_client() -> OrbitalIQClient:
    """A Streamlit-session-cached client instance (one pooled httpx.Client
    per (browser session, backend URL) pair, not one per rerun). Keyed
    explicitly by the resolved base URL — rather than caching a single
    no-argument client — so that if the configured backend URL ever
    changes (a different ``ORBITALIQ_API_BASE_URL``, e.g. via secrets or
    an env var), a stale client pointed at the old URL is never silently
    reused.
    """
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def _build(base_url: str) -> OrbitalIQClient:
        return OrbitalIQClient(base_url=base_url)

    return _build(get_api_base_url())
