"""Tests for the SQLite hardening in data/database.py — WAL journal mode
and a raised busy-timeout, added specifically to reduce the odds of an
unhandled ``sqlite3.OperationalError: database is locked`` under FastAPI's
concurrent (threaded) request execution, which is the leading root-cause
candidate identified for the historical "500: Internal Server Error"
report (see README > Known limitations).
"""
from __future__ import annotations

import pytest
from sqlalchemy import text as sqlalchemy_text

from orbitaliq.data.database import _make_engine


def test_file_backed_sqlite_engine_enables_wal_and_busy_timeout(tmp_path):
    db_path = tmp_path / "hardening_test.db"
    engine = _make_engine(f"sqlite:///{db_path}")
    try:
        with engine.connect() as conn:
            journal_mode = conn.exec_driver_sql("PRAGMA journal_mode").scalar()
            busy_timeout = conn.exec_driver_sql("PRAGMA busy_timeout").scalar()
            foreign_keys = conn.exec_driver_sql("PRAGMA foreign_keys").scalar()
        assert str(journal_mode).lower() == "wal"
        assert int(busy_timeout) == 30000
        assert int(foreign_keys) == 1
    finally:
        engine.dispose()


def test_file_backed_sqlite_engine_passes_a_generous_busy_timeout_to_the_driver(tmp_path, monkeypatch):
    """``connect_args={"timeout": 30}`` is sqlite3's own DBAPI-level busy
    timeout (in seconds) — how long a connection waits to acquire a lock
    before raising ``database is locked`` — independent of the WAL-mode
    ``PRAGMA busy_timeout`` set on every new connection. Verified
    deterministically by capturing the exact arguments ``create_engine``
    is called with, rather than racing real threads against a real file
    lock (which would make this test's pass/fail depend on OS scheduling
    timing).
    """
    import sqlalchemy

    captured = {}
    real_create_engine = sqlalchemy.create_engine

    def _spy_create_engine(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return real_create_engine(url, **kwargs)

    monkeypatch.setattr("orbitaliq.data.database.create_engine", _spy_create_engine)

    db_path = tmp_path / "hardening_timeout_test.db"
    engine = _make_engine(f"sqlite:///{db_path}")
    try:
        assert captured["kwargs"]["connect_args"] == {"check_same_thread": False, "timeout": 30}
    finally:
        engine.dispose()


def test_get_db_yields_a_working_session_and_closes_it(tmp_path, monkeypatch):
    import sqlalchemy.orm as orm

    from orbitaliq.data import database as database_module

    engine = _make_engine(f"sqlite:///{tmp_path / 'get_db_test.db'}")
    from orbitaliq.data.models import Base

    Base.metadata.create_all(engine)
    test_session_local = orm.sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(database_module, "SessionLocal", test_session_local)

    gen = database_module.get_db()
    session = next(gen)
    assert session.execute(sqlalchemy_text("SELECT 1")).scalar() == 1
    gen.close()  # triggers the generator's `finally: session.close()`
    engine.dispose()


def test_session_scope_commits_on_success_and_rolls_back_on_error(tmp_path):
    import sqlalchemy.orm as orm

    from orbitaliq.data.database import session_scope
    from orbitaliq.data.models import AssessmentRecord, Base

    engine = _make_engine(f"sqlite:///{tmp_path / 'session_scope_test.db'}")
    Base.metadata.create_all(engine)
    session_factory = orm.sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    def _make_record(ticker: str) -> AssessmentRecord:
        return AssessmentRecord(
            ticker=ticker, company_name="X", facility_name="Y", latitude=1.0, longitude=1.0,
            momentum_score=1.0, momentum_level="STABLE", narrative_report="n",
        )

    with session_scope(session_factory) as session:
        session.add(_make_record("COMMIT_OK"))

    with session_factory() as verify_session:
        assert verify_session.query(AssessmentRecord).filter_by(ticker="COMMIT_OK").count() == 1

    with pytest.raises(RuntimeError):
        with session_scope(session_factory) as session:
            session.add(_make_record("SHOULD_ROLL_BACK"))
            raise RuntimeError("simulated failure mid-transaction")

    with session_factory() as verify_session:
        assert verify_session.query(AssessmentRecord).filter_by(ticker="SHOULD_ROLL_BACK").count() == 0

    engine.dispose()


def test_make_engine_is_idempotent_and_tables_still_creatable(tmp_path):
    from orbitaliq.data.models import Base

    db_path = tmp_path / "hardening_create_all.db"
    engine = _make_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with engine.connect() as conn:
            tables = conn.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).scalars().all()
        assert "assessments" in tables
        assert "watchlist_entries" in tables
    finally:
        engine.dispose()


# --- Legacy-schema migration (regression coverage for the confirmed root
# cause of "500 Internal Server Error" on Executive Dashboard / Company
# Intelligence / Assessment List: a real project database was inspected
# directly with PRAGMA table_info and found to be missing all 11 columns
# AssessmentRecord has grown since that database file was first created —
# Base.metadata.create_all() never alters an existing table, so every ORM
# read/write against it raised sqlite3.OperationalError: no such column) ---

_LEGACY_ASSESSMENTS_TABLE_DDL = """
CREATE TABLE assessments (
    id VARCHAR(36) NOT NULL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    facility_name VARCHAR(255) NOT NULL,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    industry VARCHAR(120) NOT NULL,
    market_cap_usd FLOAT,
    momentum_score FLOAT NOT NULL,
    momentum_level VARCHAR(30) NOT NULL,
    signal_breakdown JSON NOT NULL,
    vision_findings JSON NOT NULL,
    recommended_actions JSON NOT NULL,
    narrative_report VARCHAR NOT NULL,
    agent_trace JSON NOT NULL,
    llm_generated BOOLEAN NOT NULL,
    data_sources JSON NOT NULL,
    created_at DATETIME NOT NULL
)
"""

_LEGACY_ROW = {
    "id": "4c94550f-ce25-4693-bc28-497e9de84abe",
    "ticker": "TSLA",
    "company_name": "Tesla, Inc.",
    "facility_name": "Gigafactory Nevada",
    "latitude": 39.538,
    "longitude": -119.4425,
    "industry": "automotive",
    "market_cap_usd": 800000000000.0,
    "momentum_score": 35.91735720018142,
    "momentum_level": "EMERGING",
    "signal_breakdown": '{"revenue_growth": 17.67}',
    "vision_findings": '{"overall_change_score": 0.84}',
    "recommended_actions": "[]",
    "narrative_report": "Tesla, Inc. (TSLA) carries a composite momentum score of 35.9/100.",
    "agent_trace": "[]",
    "llm_generated": 0,
    "data_sources": '{"revenue_growth_signal": "sec_edgar_live"}',
    "created_at": "2026-09-18 15:58:42.514393",
}


def _create_legacy_database(db_path) -> "sqlalchemy.engine.Engine":
    import sqlalchemy

    engine = sqlalchemy.create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        conn.exec_driver_sql(_LEGACY_ASSESSMENTS_TABLE_DDL)
        columns = ", ".join(_LEGACY_ROW.keys())
        placeholders = ", ".join(f":{k}" for k in _LEGACY_ROW.keys())
        conn.exec_driver_sql(f"INSERT INTO assessments ({columns}) VALUES ({placeholders})", _LEGACY_ROW)
    return engine


def test_migrate_sqlite_adds_every_missing_enterprise_column_to_a_legacy_table(tmp_path):
    from orbitaliq.data.database import _ASSESSMENT_ADDITIVE_COLUMNS, _migrate_sqlite_add_missing_columns
    from sqlalchemy import inspect as sa_inspect

    engine = _create_legacy_database(tmp_path / "legacy.db")
    try:
        before = {c["name"] for c in sa_inspect(engine).get_columns("assessments")}
        assert not set(_ASSESSMENT_ADDITIVE_COLUMNS).intersection(before), "fixture should start on the old schema"

        added = _migrate_sqlite_add_missing_columns(engine)

        assert set(added) == set(_ASSESSMENT_ADDITIVE_COLUMNS)
        after = {c["name"] for c in sa_inspect(engine).get_columns("assessments")}
        assert set(_ASSESSMENT_ADDITIVE_COLUMNS).issubset(after)
    finally:
        engine.dispose()


def test_migrate_sqlite_preserves_existing_row_data_exactly(tmp_path):
    from orbitaliq.data.database import _migrate_sqlite_add_missing_columns

    engine = _create_legacy_database(tmp_path / "legacy_preserve.db")
    try:
        _migrate_sqlite_add_missing_columns(engine)
        with engine.connect() as conn:
            row = conn.exec_driver_sql(
                "SELECT id, ticker, momentum_score, narrative_report FROM assessments WHERE id = :id",
                {"id": _LEGACY_ROW["id"]},
            ).mappings().one()
        assert row["ticker"] == "TSLA"
        assert row["momentum_score"] == _LEGACY_ROW["momentum_score"]
        assert row["narrative_report"] == _LEGACY_ROW["narrative_report"]
    finally:
        engine.dispose()


def test_migrate_sqlite_gives_sensible_defaults_for_the_migrated_row(tmp_path):
    """The one real row a legacy database might already contain must load
    through the *current* ORM model without raising, and the newly-added
    columns must come back as the same honest "nothing computed yet"
    defaults a fresh assessment would get — never null-as-if-fabricated.
    """
    import sqlalchemy.orm as orm

    from orbitaliq.data.database import _migrate_sqlite_add_missing_columns
    from orbitaliq.data.models import AssessmentRecord

    engine = _create_legacy_database(tmp_path / "legacy_defaults.db")
    try:
        _migrate_sqlite_add_missing_columns(engine)
        session_factory = orm.sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
        with session_factory() as session:
            record = session.get(AssessmentRecord, _LEGACY_ROW["id"])
        assert record is not None
        assert record.financial_profile is None
        assert record.convergence is None
        assert record.data_quality is None
        assert record.evidence == []
        assert record.narrative_status == "DETERMINISTIC_GROUNDED_TEMPLATE"
        assert record.narrative_grounding is None
        assert record.satellite_visual is None
        assert record.watchlist_tier == "NONE"
        assert record.watchlist_webhook_status == "NOT_APPLICABLE"
        assert record.watchlist_flagged is False
        # to_dict() is what the API response layer calls — must not raise.
        as_dict = record.to_dict()
        assert as_dict["ticker"] == "TSLA"
    finally:
        engine.dispose()


def test_migrate_sqlite_is_idempotent(tmp_path):
    from orbitaliq.data.database import _migrate_sqlite_add_missing_columns

    engine = _create_legacy_database(tmp_path / "legacy_idempotent.db")
    try:
        first_pass = _migrate_sqlite_add_missing_columns(engine)
        second_pass = _migrate_sqlite_add_missing_columns(engine)
        assert first_pass  # something was actually added the first time
        assert second_pass == []  # nothing left to add the second time — no "duplicate column" error
    finally:
        engine.dispose()


def test_migrate_sqlite_is_a_noop_on_a_brand_new_database(tmp_path):
    """A database created fresh by today's code already has every column
    via create_all() — the migration must not try to re-add them.
    """
    from orbitaliq.data.database import _migrate_sqlite_add_missing_columns
    from orbitaliq.data.models import Base

    engine = _make_engine(f"sqlite:///{tmp_path / 'brand_new.db'}")
    try:
        Base.metadata.create_all(engine)
        added = _migrate_sqlite_add_missing_columns(engine)
        assert added == []
    finally:
        engine.dispose()


def test_migrate_sqlite_is_a_noop_when_the_table_does_not_exist_yet(tmp_path):
    from orbitaliq.data.database import _migrate_sqlite_add_missing_columns

    engine = _make_engine(f"sqlite:///{tmp_path / 'no_table_yet.db'}")
    try:
        added = _migrate_sqlite_add_missing_columns(engine)
        assert added == []
    finally:
        engine.dispose()


# --- _migrate_narrative_status_labels: honest relabeling of a since-renamed value ---


def _legacy_database_with_old_narrative_status_label(tmp_path, filename):
    """A database that already has the modern schema (via the additive-
    column migration) but still carries a row persisted under the old,
    pre-rename narrative_status label -- exactly what a real database that
    was migrated once, before this relabel shipped, looks like.
    """
    from orbitaliq.data.database import _migrate_sqlite_add_missing_columns

    engine = _create_legacy_database(tmp_path / filename)
    _migrate_sqlite_add_missing_columns(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE assessments SET narrative_status = 'DETERMINISTIC_OFFLINE_TEMPLATE' WHERE id = :id",
            {"id": _LEGACY_ROW["id"]},
        )
    return engine


def test_migrate_narrative_status_labels_relabels_the_old_value(tmp_path):
    from orbitaliq.data.database import _migrate_narrative_status_labels

    engine = _legacy_database_with_old_narrative_status_label(tmp_path, "legacy_narrative_status.db")
    try:
        updated = _migrate_narrative_status_labels(engine)
        assert updated == 1
        with engine.connect() as conn:
            row = conn.exec_driver_sql(
                "SELECT narrative_status FROM assessments WHERE id = :id", {"id": _LEGACY_ROW["id"]}
            ).mappings().one()
        assert row["narrative_status"] == "DETERMINISTIC_GROUNDED_TEMPLATE"
    finally:
        engine.dispose()


def test_migrate_narrative_status_labels_never_touches_a_different_value(tmp_path):
    """A row already carrying any other narrative_status (e.g. a genuine
    AI_GENERATED_GROUNDED assessment) must be left completely untouched --
    this migration is a pure, narrow relabel of one specific old string,
    never a blanket rewrite.
    """
    from orbitaliq.data.database import _migrate_narrative_status_labels, _migrate_sqlite_add_missing_columns

    engine = _create_legacy_database(tmp_path / "legacy_other_status.db")
    _migrate_sqlite_add_missing_columns(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE assessments SET narrative_status = 'AI_GENERATED_GROUNDED' WHERE id = :id",
            {"id": _LEGACY_ROW["id"]},
        )
    try:
        updated = _migrate_narrative_status_labels(engine)
        assert updated == 0
        with engine.connect() as conn:
            row = conn.exec_driver_sql(
                "SELECT narrative_status FROM assessments WHERE id = :id", {"id": _LEGACY_ROW["id"]}
            ).mappings().one()
        assert row["narrative_status"] == "AI_GENERATED_GROUNDED"
    finally:
        engine.dispose()


def test_migrate_narrative_status_labels_is_idempotent(tmp_path):
    from orbitaliq.data.database import _migrate_narrative_status_labels

    engine = _legacy_database_with_old_narrative_status_label(tmp_path, "legacy_narrative_idempotent.db")
    try:
        first_pass = _migrate_narrative_status_labels(engine)
        second_pass = _migrate_narrative_status_labels(engine)
        assert first_pass == 1
        assert second_pass == 0  # already relabeled -- nothing left to do
    finally:
        engine.dispose()


def test_migrate_narrative_status_labels_is_a_noop_when_the_table_does_not_exist_yet(tmp_path):
    from orbitaliq.data.database import _migrate_narrative_status_labels

    engine = _make_engine(f"sqlite:///{tmp_path / 'no_table_yet_2.db'}")
    try:
        assert _migrate_narrative_status_labels(engine) == 0
    finally:
        engine.dispose()


def test_init_db_relabels_a_legacy_narrative_status_end_to_end(tmp_path):
    """The end-to-end guarantee, mirroring
    test_init_db_repairs_a_legacy_database_so_real_api_queries_would_succeed
    above: calling init_db (exactly what main.py's startup does) against a
    database with the old narrative_status label relabels it, so the
    dashboard's badge logic (which only recognizes the new label) renders
    correctly for old, already-persisted assessments too.
    """
    import sqlalchemy.orm as orm

    from orbitaliq.data.database import init_db
    from orbitaliq.data.models import AssessmentRecord

    engine = _legacy_database_with_old_narrative_status_label(tmp_path, "legacy_init_db_narrative.db")
    try:
        init_db(bind_engine=engine)
        session_factory = orm.sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
        with session_factory() as session:
            record = session.get(AssessmentRecord, _LEGACY_ROW["id"])
        assert record is not None
        assert record.narrative_status == "DETERMINISTIC_GROUNDED_TEMPLATE"
    finally:
        engine.dispose()


# --- _migrate_sqlite_relax_lat_lon_not_null: financial-only assessments (no
# verified facility coordinates -- see data/company_resolver.py) need to
# store a genuine NULL latitude/longitude rather than ever being forced to
# fabricate a placeholder coordinate to satisfy an old NOT NULL schema. ---


def test_migrate_relax_lat_lon_removes_not_null_from_a_legacy_table(tmp_path):
    from sqlalchemy import inspect as sa_inspect

    from orbitaliq.data.database import _migrate_sqlite_add_missing_columns, _migrate_sqlite_relax_lat_lon_not_null

    engine = _create_legacy_database(tmp_path / "legacy_relax.db")
    try:
        _migrate_sqlite_add_missing_columns(engine)  # must run first, exactly as init_db() sequences it
        before = {c["name"]: c for c in sa_inspect(engine).get_columns("assessments")}
        assert before["latitude"]["nullable"] is False
        assert before["longitude"]["nullable"] is False

        changed = _migrate_sqlite_relax_lat_lon_not_null(engine)

        assert changed is True
        after = {c["name"]: c for c in sa_inspect(engine).get_columns("assessments")}
        assert after["latitude"]["nullable"] is True
        assert after["longitude"]["nullable"] is True
    finally:
        engine.dispose()


def test_migrate_relax_lat_lon_preserves_every_existing_row_verbatim(tmp_path):
    from orbitaliq.data.database import _migrate_sqlite_add_missing_columns, _migrate_sqlite_relax_lat_lon_not_null

    engine = _create_legacy_database(tmp_path / "legacy_relax_preserve.db")
    try:
        _migrate_sqlite_add_missing_columns(engine)
        _migrate_sqlite_relax_lat_lon_not_null(engine)
        with engine.connect() as conn:
            row = conn.exec_driver_sql(
                "SELECT id, ticker, latitude, longitude, momentum_score, narrative_report FROM assessments WHERE id = :id",
                {"id": _LEGACY_ROW["id"]},
            ).mappings().one()
        assert row["ticker"] == "TSLA"
        assert row["latitude"] == pytest.approx(_LEGACY_ROW["latitude"])
        assert row["longitude"] == pytest.approx(_LEGACY_ROW["longitude"])
        assert row["momentum_score"] == _LEGACY_ROW["momentum_score"]
        assert row["narrative_report"] == _LEGACY_ROW["narrative_report"]
    finally:
        engine.dispose()


def test_migrate_relax_lat_lon_lets_a_new_row_store_null_coordinates(tmp_path):
    """The whole point of the migration: after it runs, a genuinely
    financial-only assessment (no verified facility -- see
    data/company_resolver.py's RESOLVED_NO_FACILITY status) can be persisted
    with latitude=None/longitude=None without SQLite raising a NOT NULL
    constraint failure.
    """
    import sqlalchemy.orm as orm

    from orbitaliq.data.database import _migrate_sqlite_add_missing_columns, _migrate_sqlite_relax_lat_lon_not_null
    from orbitaliq.data.models import AssessmentRecord

    engine = _create_legacy_database(tmp_path / "legacy_relax_new_row.db")
    try:
        _migrate_sqlite_add_missing_columns(engine)
        _migrate_sqlite_relax_lat_lon_not_null(engine)
        session_factory = orm.sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
        with session_factory() as session:
            record = AssessmentRecord(
                ticker="NOFAC", company_name="No Facility Co", facility_name="No verified facility",
                latitude=None, longitude=None, momentum_score=1.0, momentum_level="STABLE", narrative_report="n",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            assert record.latitude is None
            assert record.longitude is None
    finally:
        engine.dispose()


def test_migrate_relax_lat_lon_is_idempotent(tmp_path):
    from orbitaliq.data.database import _migrate_sqlite_add_missing_columns, _migrate_sqlite_relax_lat_lon_not_null

    engine = _create_legacy_database(tmp_path / "legacy_relax_idempotent.db")
    try:
        _migrate_sqlite_add_missing_columns(engine)
        first_pass = _migrate_sqlite_relax_lat_lon_not_null(engine)
        second_pass = _migrate_sqlite_relax_lat_lon_not_null(engine)
        assert first_pass is True
        assert second_pass is False  # already migrated -- no-op, no "table already exists" error
    finally:
        engine.dispose()


def test_migrate_relax_lat_lon_is_a_noop_on_a_brand_new_database(tmp_path):
    """A database created fresh by today's code already has nullable
    latitude/longitude via create_all() -- the migration must not attempt
    (and does not need) a rebuild.
    """
    from orbitaliq.data.database import _migrate_sqlite_relax_lat_lon_not_null
    from orbitaliq.data.models import Base

    engine = _make_engine(f"sqlite:///{tmp_path / 'brand_new_relax.db'}")
    try:
        Base.metadata.create_all(engine)
        assert _migrate_sqlite_relax_lat_lon_not_null(engine) is False
    finally:
        engine.dispose()


def test_migrate_relax_lat_lon_is_a_noop_when_the_table_does_not_exist_yet(tmp_path):
    from orbitaliq.data.database import _migrate_sqlite_relax_lat_lon_not_null

    engine = _make_engine(f"sqlite:///{tmp_path / 'no_table_yet_relax.db'}")
    try:
        assert _migrate_sqlite_relax_lat_lon_not_null(engine) is False
    finally:
        engine.dispose()


def test_init_db_relaxes_lat_lon_end_to_end_and_accepts_a_null_coordinate_row(tmp_path):
    """The end-to-end guarantee, mirroring the other init_db regression
    tests: calling init_db (exactly what main.py's startup does) against a
    real legacy database file leaves it able to store a financial-only
    assessment with no verified facility coordinates.
    """
    import sqlalchemy.orm as orm

    from orbitaliq.data.database import init_db
    from orbitaliq.data.models import AssessmentRecord

    engine = _create_legacy_database(tmp_path / "legacy_init_db_relax.db")
    try:
        init_db(bind_engine=engine)
        session_factory = orm.sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
        with session_factory() as session:
            # The pre-existing legacy row must still be there, untouched.
            existing = session.get(AssessmentRecord, _LEGACY_ROW["id"])
            assert existing is not None
            assert existing.latitude == pytest.approx(_LEGACY_ROW["latitude"])

            # And a brand-new financial-only row (no coordinates) must now
            # be storable without a NOT NULL failure.
            record = AssessmentRecord(
                ticker="NOFAC2", company_name="No Facility Co 2", facility_name="No verified facility",
                latitude=None, longitude=None, momentum_score=1.0, momentum_level="STABLE", narrative_report="n",
            )
            session.add(record)
            session.commit()
    finally:
        engine.dispose()


def test_init_db_repairs_a_legacy_database_so_real_api_queries_would_succeed(tmp_path):
    """The end-to-end guarantee: calling ``init_db`` (exactly what
    ``main.py``'s startup does) against a legacy-schema database file makes
    it fully queryable again — this is what fixes the reported 500s, not
    just a smaller unit-level fact about one helper function.
    """
    import sqlalchemy.orm as orm

    from orbitaliq.data.database import init_db
    from orbitaliq.data.models import AssessmentRecord

    engine = _create_legacy_database(tmp_path / "legacy_init_db.db")
    try:
        init_db(bind_engine=engine)  # exactly what create_app()'s lifespan calls
        session_factory = orm.sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
        with session_factory() as session:
            all_rows = session.query(AssessmentRecord).all()  # this is the exact call list_assessments makes
            one_row = session.get(AssessmentRecord, _LEGACY_ROW["id"])  # what get_assessment makes
        assert len(all_rows) == 1
        assert one_row is not None
        assert one_row.to_dict()["momentum_score"] == _LEGACY_ROW["momentum_score"]
    finally:
        engine.dispose()
