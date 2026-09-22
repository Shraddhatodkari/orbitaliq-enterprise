import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orbitaliq.data.models import AssessmentRecord, Base
from orbitaliq.data.repository import AssessmentRepository, WatchlistRepository, lookup_cached_financial_value


def _record(**overrides) -> AssessmentRecord:
    defaults = dict(
        ticker="TSLA",
        company_name="Tesla, Inc.",
        facility_name="Gigafactory Nevada",
        latitude=1.0,
        longitude=2.0,
        industry="automotive",
        market_cap_usd=None,
        momentum_score=42.0,
        momentum_level="EMERGING",
        signal_breakdown={"revenue_growth": 42.0},
        vision_findings={},
        recommended_actions=["do something"],
        narrative_report="a report",
        agent_trace=[],
        llm_generated=False,
    )
    defaults.update(overrides)
    return AssessmentRecord(**defaults)


def test_add_and_get_round_trip(db_session):
    repo = AssessmentRepository(db_session)
    record = repo.add(_record())
    db_session.commit()

    fetched = repo.get(record.id)
    assert fetched is not None
    assert fetched.company_name == "Tesla, Inc."
    assert fetched.momentum_score == 42.0


def test_get_missing_returns_none(db_session):
    repo = AssessmentRepository(db_session)
    assert repo.get("does-not-exist") is None


def test_list_orders_newest_first_and_respects_limit(db_session):
    repo = AssessmentRepository(db_session)
    for i in range(5):
        repo.add(_record(company_name=f"Company {i}", momentum_score=float(i)))
    db_session.commit()

    results = repo.list(limit=3)
    assert len(results) == 3
    assert repo.count() == 5


def test_list_filters_by_min_momentum_score(db_session):
    repo = AssessmentRepository(db_session)
    repo.add(_record(company_name="Low", momentum_score=10.0))
    repo.add(_record(company_name="High", momentum_score=90.0))
    db_session.commit()

    results = repo.list(min_momentum_score=50.0)
    assert len(results) == 1
    assert results[0].company_name == "High"


def test_to_dict_contains_all_expected_keys(db_session):
    repo = AssessmentRepository(db_session)
    record = repo.add(_record())
    db_session.commit()
    d = record.to_dict()
    for key in [
        "id",
        "ticker",
        "company_name",
        "facility_name",
        "latitude",
        "longitude",
        "industry",
        "momentum_score",
        "momentum_level",
        "signal_breakdown",
        "vision_findings",
        "recommended_actions",
        "narrative_report",
        "agent_trace",
        "llm_generated",
        "data_sources",
        "watchlist_flagged",
        "watchlist_channel",
        "watchlist_tier",
        "watchlist_webhook_status",
        "financial_profile",
        "convergence",
        "data_quality",
        "evidence",
        "narrative_status",
        "narrative_grounding",
        "satellite_visual",
        "created_at",
    ]:
        assert key in d


# --- WatchlistRepository ---


def test_watchlist_upsert_creates_new_entry(db_session):
    record = AssessmentRepository(db_session).add(
        _record(ticker="HOT", company_name="Hot Corp", facility_name="Big Site", momentum_score=90.0, momentum_level="AGGRESSIVE_EXPANSION")
    )
    record.watchlist_tier = "HIGH_SIGNAL"
    record.watchlist_channel = "competitive-intel-priority"
    record.watchlist_webhook_status = "STUBBED_NO_WEBHOOK_CONFIGURED"
    db_session.flush()

    entry = WatchlistRepository(db_session).upsert_from_assessment(record)
    db_session.commit()

    assert entry.ticker == "HOT"
    assert entry.tier == "HIGH_SIGNAL"
    assert entry.latest_assessment_id == record.id
    assert entry.first_flagged_at is not None


def test_watchlist_upsert_updates_existing_entry_for_same_ticker_and_facility(db_session):
    repo = AssessmentRepository(db_session)
    wl_repo = WatchlistRepository(db_session)

    first = repo.add(_record(ticker="HOT", facility_name="Big Site", momentum_score=40.0, momentum_level="EMERGING"))
    first.watchlist_tier = "WATCH"
    db_session.flush()
    entry1 = wl_repo.upsert_from_assessment(first)
    db_session.commit()

    second = repo.add(_record(ticker="HOT", facility_name="Big Site", momentum_score=90.0, momentum_level="AGGRESSIVE_EXPANSION"))
    second.watchlist_tier = "HIGH_SIGNAL"
    db_session.flush()
    entry2 = wl_repo.upsert_from_assessment(second)
    db_session.commit()

    # Same row updated in place, not a second row.
    assert entry1.id == entry2.id
    assert entry2.tier == "HIGH_SIGNAL"
    assert entry2.momentum_score == 90.0
    assert entry2.latest_assessment_id == second.id


def test_watchlist_upsert_different_facility_creates_separate_entry(db_session):
    repo = AssessmentRepository(db_session)
    wl_repo = WatchlistRepository(db_session)

    a = repo.add(_record(ticker="HOT", facility_name="Site A", momentum_score=90.0, momentum_level="AGGRESSIVE_EXPANSION"))
    a.watchlist_tier = "HIGH_SIGNAL"
    db_session.flush()
    wl_repo.upsert_from_assessment(a)

    b = repo.add(_record(ticker="HOT", facility_name="Site B", momentum_score=90.0, momentum_level="AGGRESSIVE_EXPANSION"))
    b.watchlist_tier = "HIGH_SIGNAL"
    db_session.flush()
    wl_repo.upsert_from_assessment(b)
    db_session.commit()

    assert len(wl_repo.list()) == 2


def test_watchlist_list_defaults_to_excluding_none_tier(db_session):
    repo = AssessmentRepository(db_session)
    wl_repo = WatchlistRepository(db_session)

    stable = repo.add(_record(ticker="CALM", momentum_score=5.0, momentum_level="STABLE"))
    stable.watchlist_tier = "NONE"
    db_session.flush()
    wl_repo.upsert_from_assessment(stable)

    hot = repo.add(_record(ticker="HOT", momentum_score=90.0, momentum_level="AGGRESSIVE_EXPANSION"))
    hot.watchlist_tier = "HIGH_SIGNAL"
    db_session.flush()
    wl_repo.upsert_from_assessment(hot)
    db_session.commit()

    default_view = wl_repo.list()
    assert {e.ticker for e in default_view} == {"HOT"}

    # tier="NONE" is not an equality filter -- it means "no filter", i.e.
    # every entry ever assessed, watchlisted or not (see the dedicated
    # regression test below for this specific contract).
    everyone = wl_repo.list(tier="NONE")
    assert {e.ticker for e in everyone} == {"CALM", "HOT"}


def test_watchlist_list_filters_by_tier_and_sorts_by_momentum_desc(db_session):
    repo = AssessmentRepository(db_session)
    wl_repo = WatchlistRepository(db_session)

    for ticker, score, level, tier in [
        ("LOW", 40.0, "EMERGING", "WATCH"),
        ("MED", 60.0, "STRONG", "REVIEW"),
        ("HIGH", 90.0, "AGGRESSIVE_EXPANSION", "HIGH_SIGNAL"),
    ]:
        rec = repo.add(_record(ticker=ticker, momentum_score=score, momentum_level=level))
        rec.watchlist_tier = tier
        db_session.flush()
        wl_repo.upsert_from_assessment(rec)
    db_session.commit()

    all_entries = wl_repo.list()
    assert [e.ticker for e in all_entries] == ["HIGH", "MED", "LOW"]  # sorted by momentum_score desc

    review_only = wl_repo.list(tier="REVIEW")
    assert [e.ticker for e in review_only] == ["MED"]


def test_watchlist_list_tier_none_returns_everyone_not_just_deescalated(db_session):
    """Regression test: tier='NONE' means "no filter, show every entry
    ever assessed (including de-escalated)" -- it must NOT be treated as
    an ordinary equality filter that would return only rows whose tier
    literally equals 'NONE'.
    """
    repo = AssessmentRepository(db_session)
    wl_repo = WatchlistRepository(db_session)

    watched = repo.add(_record(ticker="HOT", momentum_score=90.0, momentum_level="AGGRESSIVE_EXPANSION"))
    watched.watchlist_tier = "HIGH_SIGNAL"
    db_session.flush()
    wl_repo.upsert_from_assessment(watched)

    calm = repo.add(_record(ticker="CALM", momentum_score=5.0, momentum_level="STABLE"))
    calm.watchlist_tier = "NONE"
    db_session.flush()
    wl_repo.upsert_from_assessment(calm)
    db_session.commit()

    everyone = wl_repo.list(tier="NONE")
    assert {e.ticker for e in everyone} == {"HOT", "CALM"}


# --- lookup_cached_financial_value: genuine CACHED_REAL support ---


@pytest.fixture
def patched_session_local(monkeypatch):
    """lookup_cached_financial_value() deliberately opens its own
    short-lived session via orbitaliq.data.database.SessionLocal (so it
    can be injected as a plain callable into IngestionAgent without
    threading a request-scoped session through) -- point that at a fresh,
    isolated in-memory database for this test rather than whatever
    process-wide database.py's module-level engine happens to be bound to.
    """
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr("orbitaliq.data.database.SessionLocal", session_factory)
    yield session_factory
    engine.dispose()


def _insert_record(session_factory, **overrides) -> None:
    session = session_factory()
    try:
        session.add(_record(**overrides))
        session.commit()
    finally:
        session.close()


def test_lookup_cached_financial_value_returns_most_recent_trusted_value(patched_session_local):
    _insert_record(patched_session_local, ticker="TSLA", signal_breakdown={"revenue_growth": 55.0})
    result = lookup_cached_financial_value("TSLA", "revenue_growth_signal")
    assert result is not None
    value, retrieved_at = result
    assert value == pytest.approx(0.55)  # signal_breakdown is 0-100 scaled; the raw signal value is 0-1
    assert retrieved_at is not None


def test_lookup_cached_financial_value_is_case_insensitive_on_ticker(patched_session_local):
    _insert_record(patched_session_local, ticker="tsla", signal_breakdown={"capex_growth": 30.0})
    result = lookup_cached_financial_value("TSLA", "capex_growth_signal")
    assert result is not None
    assert result[0] == pytest.approx(0.30)


def test_lookup_cached_financial_value_none_on_clean_miss(patched_session_local):
    # No records at all for this ticker.
    assert lookup_cached_financial_value("NOPE", "revenue_growth_signal") is None


def test_lookup_cached_financial_value_none_when_signal_never_persisted(patched_session_local):
    # A record exists for this ticker, but it never had a trusted
    # revenue_growth value (e.g. that signal was itself excluded at the
    # time), so signal_breakdown simply doesn't have that key.
    _insert_record(patched_session_local, ticker="TSLA", signal_breakdown={"capex_growth": 10.0})
    assert lookup_cached_financial_value("TSLA", "revenue_growth_signal") is None


def test_lookup_cached_financial_value_unknown_signal_key_returns_none(patched_session_local):
    _insert_record(patched_session_local, ticker="TSLA", signal_breakdown={"revenue_growth": 50.0})
    assert lookup_cached_financial_value("TSLA", "not_a_real_signal") is None


def test_lookup_cached_financial_value_never_raises_when_database_unavailable(monkeypatch):
    """A DB error (or SessionLocal not even pointing at a real database)
    must degrade to a clean None, never propagate and block an assessment.
    """

    def _broken_session_factory():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr("orbitaliq.data.database.SessionLocal", _broken_session_factory)
    assert lookup_cached_financial_value("TSLA", "revenue_growth_signal") is None
