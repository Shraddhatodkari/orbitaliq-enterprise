from __future__ import annotations

from orbitaliq.core.evidence import build_evidence_trail


def _available_metric(**overrides):
    defaults = dict(
        key="revenue_growth", label="Revenue Growth", category="growth", unit="percent", status="AVAILABLE",
        current_value=0.284, prior_value=0.10, yoy_change_pct=0.284,
        fiscal_period="FY2024", prior_fiscal_period="FY2023", xbrl_concept="Revenues", source="sec_edgar_live",
        note=None,
    )
    defaults.update(overrides)
    return defaults


def test_available_financial_metrics_produce_evidence_items_with_calculation():
    items = build_evidence_trail(
        company_name="Tesla, Inc.", ticker="TSLA", facility_name="Gigafactory Nevada",
        financial_metrics={"revenue_growth": _available_metric()},
        filing_source_url="https://data.sec.gov/x",
        financial_retrieved_at="2026-09-18T00:00:00Z",
        satellite_meta={}, vision_findings={}, momentum_score=50.0, momentum_level="EMERGING",
        signal_breakdown={},
    )
    financial_items = [i for i in items if i.dashboard_signal == "financial_profile.revenue_growth"]
    assert len(financial_items) == 1
    item = financial_items[0]
    assert item.source_name == "SEC EDGAR XBRL Company Facts"
    assert item.source_url == "https://data.sec.gov/x"
    assert "FY2024" in item.document
    assert "Revenues" in item.document
    assert "+28.40%" in item.calculation


def test_insufficient_metrics_are_excluded_from_evidence_trail():
    items = build_evidence_trail(
        company_name="X", ticker="X", facility_name="Y",
        financial_metrics={"rd_growth": {"status": "INSUFFICIENT_DATA", "label": "R&D Growth"}},
        filing_source_url=None, financial_retrieved_at=None,
        satellite_meta={}, vision_findings={}, momentum_score=0.0, momentum_level="STABLE",
        signal_breakdown={},
    )
    assert items == [
        i for i in items if i.dashboard_signal != "financial_profile.rd_growth"
    ]
    assert all("rd_growth" not in i.dashboard_signal for i in items)


def test_metric_without_prior_period_reports_no_comparable_prior():
    metric = _available_metric(yoy_change_pct=None, prior_value=None, prior_fiscal_period=None)
    items = build_evidence_trail(
        company_name="X", ticker="X", facility_name="Y",
        financial_metrics={"revenue_growth": metric},
        filing_source_url=None, financial_retrieved_at=None,
        satellite_meta={}, vision_findings={}, momentum_score=0.0, momentum_level="STABLE",
        signal_breakdown={},
    )
    item = next(i for i in items if i.dashboard_signal == "financial_profile.revenue_growth")
    assert "no comparable prior period" in item.calculation.lower()


def test_satellite_meta_produces_one_evidence_item_when_present():
    items = build_evidence_trail(
        company_name="X", ticker="X", facility_name="Facility",
        financial_metrics={}, filing_source_url=None, financial_retrieved_at=None,
        satellite_meta={
            "source_name": "NASA GIBS (VIIRS/MODIS)", "source_url": "https://gibs.earthdata.nasa.gov/wmts/",
            "current_date": "2026-09-01", "prior_date": "2026-03-01", "resolution_m_per_pixel": 900.0,
            "latitude": 39.5, "longitude": -119.4, "retrieved_at": "2026-09-18T00:00:00Z",
        },
        vision_findings={"overall_change_score": 0.42}, momentum_score=0.0, momentum_level="STABLE",
        signal_breakdown={},
    )
    sat_items = [i for i in items if i.dashboard_signal == "vision_findings.overall_change_score"]
    assert len(sat_items) == 1
    assert "Facility" in sat_items[0].claim
    assert "0.42" in sat_items[0].value


def test_satellite_meta_empty_produces_no_satellite_evidence_item():
    items = build_evidence_trail(
        company_name="X", ticker="X", facility_name="Y",
        financial_metrics={}, filing_source_url=None, financial_retrieved_at=None,
        satellite_meta={}, vision_findings={}, momentum_score=0.0, momentum_level="STABLE",
        signal_breakdown={},
    )
    assert all(i.dashboard_signal != "vision_findings.overall_change_score" for i in items)


def test_signal_breakdown_and_momentum_score_always_produce_items():
    items = build_evidence_trail(
        company_name="Acme Corp", ticker="ACME", facility_name="HQ",
        financial_metrics={}, filing_source_url=None, financial_retrieved_at=None,
        satellite_meta={}, vision_findings={}, momentum_score=61.5, momentum_level="STRONG",
        signal_breakdown={"revenue_growth": 70.0, "capex_growth": 55.0},
    )
    breakdown_items = [i for i in items if i.dashboard_signal.startswith("signal_breakdown.")]
    assert len(breakdown_items) == 2
    momentum_items = [i for i in items if i.dashboard_signal == "momentum_score"]
    assert len(momentum_items) == 1
    # Output philosophy (task #73): the qualitative Competitive Expansion
    # Signal level leads, the 0-100 composite index is disclosed supporting
    # detail -- never the other way around.
    assert "STRONG (composite index 61.5/100)" in momentum_items[0].value
    assert "Acme Corp" in momentum_items[0].claim


def test_every_evidence_item_as_dict_has_full_shape():
    items = build_evidence_trail(
        company_name="X", ticker="X", facility_name="Y",
        financial_metrics={}, filing_source_url=None, financial_retrieved_at=None,
        satellite_meta={}, vision_findings={}, momentum_score=1.0, momentum_level="STABLE",
        signal_breakdown={},
    )
    for item in items:
        d = item.as_dict()
        assert set(d.keys()) == {
            "claim", "source_name", "source_url", "document", "retrieved_at", "value", "calculation", "dashboard_signal",
            # Audit-grade identity fields (additive) -- see core/evidence.py.
            "evidence_id", "assessment_id", "correlation_id", "agent", "model",
        }


# --- Audit-grade identity: evidence_id / assessment_id / correlation_id / agent / model ---


def test_every_evidence_item_has_a_unique_freshly_generated_evidence_id():
    items = build_evidence_trail(
        company_name="Acme Corp", ticker="ACME", facility_name="HQ",
        financial_metrics={"revenue_growth": _available_metric()},
        filing_source_url=None, financial_retrieved_at=None,
        satellite_meta={
            "source_name": "NASA GIBS", "source_url": None, "current_date": "2026-09-01",
            "prior_date": "2026-03-01", "resolution_m_per_pixel": 900.0, "latitude": 1.0, "longitude": 1.0,
            "retrieved_at": None,
        },
        vision_findings={"overall_change_score": 0.1}, momentum_score=50.0, momentum_level="EMERGING",
        signal_breakdown={"revenue_growth": 70.0},
    )
    ids = [i.evidence_id for i in items]
    assert len(ids) == len(set(ids))  # every id is unique
    assert all(isinstance(i, str) and len(i) == 36 for i in ids)  # a real uuid4 string


def test_assessment_id_and_correlation_id_are_none_when_not_supplied():
    items = build_evidence_trail(
        company_name="X", ticker="X", facility_name="Y",
        financial_metrics={}, filing_source_url=None, financial_retrieved_at=None,
        satellite_meta={}, vision_findings={}, momentum_score=1.0, momentum_level="STABLE",
        signal_breakdown={},
    )
    assert all(i.assessment_id is None and i.correlation_id is None for i in items)


def test_assessment_id_and_correlation_id_are_stamped_onto_every_item_when_supplied():
    items = build_evidence_trail(
        company_name="Acme Corp", ticker="ACME", facility_name="HQ",
        financial_metrics={"revenue_growth": _available_metric()},
        filing_source_url=None, financial_retrieved_at=None,
        satellite_meta={
            "source_name": "NASA GIBS", "source_url": None, "current_date": "2026-09-01",
            "prior_date": "2026-03-01", "resolution_m_per_pixel": 900.0, "latitude": 1.0, "longitude": 1.0,
            "retrieved_at": None,
        },
        vision_findings={"overall_change_score": 0.1}, momentum_score=50.0, momentum_level="EMERGING",
        signal_breakdown={"revenue_growth": 70.0},
        assessment_id="assessment-123",
        correlation_id="request-456",
    )
    assert items  # sanity: this run actually produced items
    assert all(i.assessment_id == "assessment-123" for i in items)
    assert all(i.correlation_id == "request-456" for i in items)


def test_agent_and_model_reflect_which_component_produced_each_claim():
    items = build_evidence_trail(
        company_name="Acme Corp", ticker="ACME", facility_name="HQ",
        financial_metrics={"revenue_growth": _available_metric()},
        filing_source_url=None, financial_retrieved_at=None,
        satellite_meta={
            "source_name": "NASA GIBS", "source_url": None, "current_date": "2026-09-01",
            "prior_date": "2026-03-01", "resolution_m_per_pixel": 900.0, "latitude": 1.0, "longitude": 1.0,
            "retrieved_at": None,
        },
        vision_findings={"overall_change_score": 0.1}, momentum_score=50.0, momentum_level="EMERGING",
        signal_breakdown={"revenue_growth": 70.0},
    )
    financial_item = next(i for i in items if i.dashboard_signal == "financial_profile.revenue_growth")
    assert financial_item.agent == "ingestion_agent"
    assert financial_item.model is None  # a direct XBRL fact, not an ML output

    satellite_item = next(i for i in items if i.dashboard_signal == "vision_findings.overall_change_score")
    assert satellite_item.agent == "vision_agent"
    assert satellite_item.model == "FacilityChangeCNN"  # the one item genuinely produced by an ML model

    breakdown_item = next(i for i in items if i.dashboard_signal == "signal_breakdown.revenue_growth")
    assert breakdown_item.agent == "intelligence_scoring_agent"
    assert breakdown_item.model is None  # deterministic composite, not an ML output

    momentum_item = next(i for i in items if i.dashboard_signal == "momentum_score")
    assert momentum_item.agent == "intelligence_scoring_agent"
    assert momentum_item.model is None
