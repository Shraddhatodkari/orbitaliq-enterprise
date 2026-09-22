"""Database engine/session management.

Uses SQLAlchemy 2.0 with a connection-pooled engine. SQLite is the default
for local development and the test-suite; swapping ``DATABASE_URL`` to a
Postgres DSN is the only change needed for a production deployment.
"""
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.orm import Session, sessionmaker

from orbitaliq.config import get_settings
from orbitaliq.core.logging_config import logger
from orbitaliq.data.models import Base

# ``Base.metadata.create_all()`` (used by ``init_db`` below) only creates
# tables that do not exist yet — it never alters an *existing* table to add
# a column a newer version of a model defines. ``AssessmentRecord`` in
# ``models.py`` has grown several times (the enterprise-dashboard fields —
# ``financial_profile``, ``convergence``, ``data_quality``, ``evidence``,
# ``narrative_status``, ``narrative_grounding``, ``satellite_visual`` — and
# the watchlist fields — ``watchlist_flagged``/``_channel``/``_tier``/
# ``_webhook_status``); a SQLite database file created before one of those
# additions is silently left on the old table shape. Every read or write
# through the ORM against that table then fails with
# ``sqlite3.OperationalError: no such column: assessments.<name>`` — this
# is a *confirmed, observed* root cause of "500 Internal Server Error" on
# the Executive Dashboard / Company Intelligence / Assessment List views
# (any endpoint that SELECTs the ``assessments`` table), reproduced by
# inspecting a real project database with ``PRAGMA table_info`` and
# comparing it against ``AssessmentRecord.__table__.columns`` — not a
# hypothetical. This map is the additive, in-place repair for it.
#
# Every entry below carries an explicit SQL type + constraint fragment
# chosen to exactly match ``AssessmentRecord``'s Python-side
# nullability/default in ``models.py``, so a migrated pre-existing row
# behaves identically to one inserted fresh by today's code. SQLite
# requires any NOT NULL column added via ``ALTER TABLE ... ADD COLUMN`` on
# a non-empty table to carry a constant DEFAULT — every NOT NULL entry
# here has one.
_ASSESSMENT_ADDITIVE_COLUMNS: dict[str, str] = {
    "watchlist_flagged": "BOOLEAN NOT NULL DEFAULT 0",
    "watchlist_channel": "VARCHAR(80)",
    "watchlist_tier": "VARCHAR(20) NOT NULL DEFAULT 'NONE'",
    "watchlist_webhook_status": "VARCHAR(40) NOT NULL DEFAULT 'NOT_APPLICABLE'",
    "financial_profile": "JSON",
    "convergence": "JSON",
    "data_quality": "JSON",
    "evidence": "JSON NOT NULL DEFAULT '[]'",
    "narrative_status": "VARCHAR(60) NOT NULL DEFAULT 'DETERMINISTIC_GROUNDED_TEMPLATE'",
    "narrative_grounding": "JSON",
    "satellite_visual": "JSON",
    "score_breakdown": "JSON",
}


def _migrate_sqlite_add_missing_columns(bind_engine) -> list[str]:
    """Additively repair an existing SQLite ``assessments`` table that
    predates one or more of ``_ASSESSMENT_ADDITIVE_COLUMNS``. No-op for a
    non-SQLite backend (a real migration tool like Alembic is the correct
    answer there) and for a brand-new database (``create_all`` above
    already built the table with every column). Never touches, drops, or
    reinterprets any existing row's existing data — only appends columns.

    Returns the list of column names actually added, so this is easy to
    assert on in tests and to log on startup.
    """
    if bind_engine.dialect.name != "sqlite":
        return []

    inspector = inspect(bind_engine)
    if "assessments" not in inspector.get_table_names():
        return []

    existing_columns = {col["name"] for col in inspector.get_columns("assessments")}
    added: list[str] = []
    with bind_engine.begin() as conn:
        for column_name, ddl_fragment in _ASSESSMENT_ADDITIVE_COLUMNS.items():
            if column_name in existing_columns:
                continue
            conn.exec_driver_sql(f"ALTER TABLE assessments ADD COLUMN {column_name} {ddl_fragment}")
            added.append(column_name)

    if added:
        logger.info(
            "orbitaliq.data.database: migrated an existing SQLite database — "
            f"added missing column(s) to 'assessments': {added}. No existing rows were altered."
        )
    return added


# The old label a pre-existing row's `narrative_status` column may still
# carry, and its honest replacement (see data/models.py's narrative_status
# comment). Renamed because "OFFLINE_TEMPLATE" wrongly implied a demo/test
# stand-in -- this deterministic-template path runs in full live-data mode
# too, and is "GROUNDED" because it's built directly from already-verified
# evidence, never generated.
_OLD_NARRATIVE_STATUS_LABEL = "DETERMINISTIC_OFFLINE_TEMPLATE"
_NEW_NARRATIVE_STATUS_LABEL = "DETERMINISTIC_GROUNDED_TEMPLATE"


# Columns whose Python-side nullability changed from NOT NULL to nullable
# after they were already shipped (unlike _ASSESSMENT_ADDITIVE_COLUMNS
# above, which are brand-new columns an ADD COLUMN can repair). SQLite has
# no ALTER COLUMN, so relaxing an existing NOT NULL constraint requires
# SQLite's own documented "recreate the table" pattern:
# https://www.sqlite.org/lang_altertable.html#otheralter
_RELAXED_NOT_NULL_COLUMNS = ("latitude", "longitude")


def _migrate_sqlite_relax_lat_lon_not_null(bind_engine) -> bool:
    """Repair a pre-existing SQLite ``assessments`` table whose
    ``latitude``/``longitude`` columns still carry the old NOT NULL
    constraint, so a financial-only assessment (no verified facility
    coordinates — see ``data/company_resolver.py``) can store a genuine
    ``NULL`` instead of ever being forced to fabricate a placeholder
    coordinate to satisfy the old schema.

    Idempotent and additive-only: a no-op on a brand-new database (already
    created with the current, nullable model) or one already migrated.
    Copies every existing row verbatim into the rebuilt table — no row is
    dropped, reordered, or reinterpreted. Must run AFTER
    ``_migrate_sqlite_add_missing_columns`` (see ``init_db`` below) so the
    old table already has every column the current model defines before
    this rebuild copies it column-for-column.
    """
    if bind_engine.dialect.name != "sqlite":
        return False

    inspector = inspect(bind_engine)
    if "assessments" not in inspector.get_table_names():
        return False

    columns = {col["name"]: col for col in inspector.get_columns("assessments")}
    still_not_null = [
        c for c in _RELAXED_NOT_NULL_COLUMNS if columns.get(c) is not None and columns[c]["nullable"] is False
    ]
    if not still_not_null:
        return False  # already migrated, or a fresh DB created with the current model

    with bind_engine.begin() as conn:
        conn.exec_driver_sql("ALTER TABLE assessments RENAME TO assessments_pre_nullable_migration")
        # Recreate 'assessments' from the CURRENT model shape (nullable
        # latitude/longitude) inside the same transaction as the rename.
        Base.metadata.tables["assessments"].create(bind=conn)

        new_columns = {c.name for c in Base.metadata.tables["assessments"].columns}
        shared_columns = [c for c in columns if c in new_columns]
        col_list = ", ".join(shared_columns)
        conn.exec_driver_sql(
            f"INSERT INTO assessments ({col_list}) SELECT {col_list} FROM assessments_pre_nullable_migration"
        )
        conn.exec_driver_sql("DROP TABLE assessments_pre_nullable_migration")

    logger.info(
        "orbitaliq.data.database: migrated an existing SQLite database — rebuilt 'assessments' so "
        f"{still_not_null} allow NULL (financial-only assessments with no verified facility coordinates). "
        "All existing rows were preserved verbatim."
    )
    return True


def _migrate_narrative_status_labels(bind_engine) -> int:
    """Additively repair existing rows that still carry the old, renamed
    ``narrative_status`` label. A pure relabeling -- it changes no other
    field and never touches a row that already has the current label or
    any other value (e.g. "AI_GENERATED_GROUNDED"). Safe to call on every
    startup (a no-op once every row has been relabeled once); returns the
    number of rows updated, so this is easy to assert on in tests and log
    on startup.
    """
    if bind_engine.dialect.name != "sqlite":
        return 0

    inspector = inspect(bind_engine)
    if "assessments" not in inspector.get_table_names():
        return 0
    existing_columns = {col["name"] for col in inspector.get_columns("assessments")}
    if "narrative_status" not in existing_columns:
        return 0

    with bind_engine.begin() as conn:
        result = conn.exec_driver_sql(
            "UPDATE assessments SET narrative_status = ? WHERE narrative_status = ?",
            (_NEW_NARRATIVE_STATUS_LABEL, _OLD_NARRATIVE_STATUS_LABEL),
        )
        updated = result.rowcount or 0

    if updated:
        logger.info(
            "orbitaliq.data.database: relabeled "
            f"{updated} existing 'assessments' row(s) from narrative_status="
            f"'{_OLD_NARRATIVE_STATUS_LABEL}' to '{_NEW_NARRATIVE_STATUS_LABEL}' "
            "(no other field changed)."
        )
    return updated


def _make_engine(database_url: str | None = None):
    url = database_url or get_settings().database_url
    is_sqlite = url.startswith("sqlite")
    # FastAPI runs sync ``def`` route handlers (every route in this app) in
    # a worker thread pool, so — even with a single uvicorn process and no
    # ``--workers`` — two requests can genuinely touch SQLite at the same
    # time (e.g. the dashboard's periodic GET /assessments or /watchlist
    # poll landing while a POST /assessments is mid-flight). Python's
    # sqlite3 driver's default busy timeout is only 5s, and on Windows/NTFS
    # SQLite's file-locking has more overhead than on Linux, so a slow live
    # assessment (real SEC EDGAR + NASA GIBS network calls take several
    # seconds) can plausibly lose that race and raise an unhandled
    # ``sqlite3.OperationalError: database is locked`` — previously
    # surfaced to the browser as an undiagnosable, generic
    # "500: Internal Server Error" (see api/routes_assessment.py's now
    # fully-wrapped persistence step). Raising the busy timeout to 30s and
    # switching to WAL journal mode (readers no longer block the writer,
    # and vice versa) makes that race far less likely to matter in
    # practice, on top of the explicit, honest error handling added below.
    connect_args = {"check_same_thread": False, "timeout": 30} if is_sqlite else {}
    new_engine = create_engine(url, connect_args=connect_args, future=True)

    if is_sqlite:

        @event.listens_for(new_engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:  # pragma: no cover - exercised via engine use
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.execute("PRAGMA foreign_keys=ON")
            finally:
                cursor.close()

    return new_engine


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db(bind_engine=None) -> None:
    """Create every table that doesn't exist yet, then additively repair
    any table that does exist but predates a later column addition (see
    ``_migrate_sqlite_add_missing_columns``) or carries a since-renamed
    label value (see ``_migrate_narrative_status_labels``). All steps are
    idempotent — safe to call on every startup, including against a
    database that already has real, valuable rows in it.
    """
    target_engine = bind_engine or engine
    Base.metadata.create_all(bind=target_engine)
    _migrate_sqlite_add_missing_columns(target_engine)
    _migrate_sqlite_relax_lat_lon_not_null(target_engine)
    _migrate_narrative_status_labels(target_engine)


@contextmanager
def session_scope(session_factory=None) -> Generator[Session, None, None]:
    """Provide a transactional scope for a series of repository operations."""
    factory = session_factory or SessionLocal
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a request-scoped session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
