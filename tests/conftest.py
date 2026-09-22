from __future__ import annotations

import os

# Force safe, deterministic, network-free defaults for the entire test
# session before any application module (which reads env at import time
# via pydantic-settings) is imported.
os.environ.setdefault("ORBITALIQ_OFFLINE_MODE", "true")
os.environ.setdefault("ORBITALIQ_ENV", "test")
os.environ.setdefault("ORBITALIQ_VISION_DEVICE", "cpu")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("NVIDIA_API_KEY", "nvapi-REPLACE_ME")
# The live data clients (SEC EDGAR, NASA GIBS) are tested directly with
# mocked HTTP responses in their own test modules. Defaulting live mode
# off here keeps the rest of the suite fast, deterministic, and
# network-free, matching ORBITALIQ_OFFLINE_MODE above.
os.environ.setdefault("ORBITALIQ_LIVE_DATA_MODE", "false")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orbitaliq.data.database import init_db
from orbitaliq.data.models import Base


@pytest.fixture
def db_session():
    """A fresh in-memory SQLite database per test."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def sample_company():
    from orbitaliq.agents.types import CompanyInput

    return CompanyInput(
        ticker="TSLA",
        company_name="Tesla, Inc.",
        facility_name="Gigafactory Nevada",
        latitude=39.5380,
        longitude=-119.4425,
        industry="automotive",
        market_cap_usd=800_000_000_000,
    )
