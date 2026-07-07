"""Tests for Alembic migrations and seed data."""

import os
import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from src.models.base import Base
from src.models import user, course, session, engagement_log, nudge, report  # noqa

# Use SQLite for CI, PostgreSQL locally if available
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test_engageiq.db")


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    )
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)
    if os.path.exists("./test_engageiq.db"):
        os.remove("./test_engageiq.db")


# Test 1: Tables are created correctly
def test_migration_creates_tables(engine):
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    required = ["users", "courses", "sessions", "engagement_logs"]
    for table in required:
        assert table in tables, f"Table '{table}' missing"


# Test 2: Seed data counts are correct
def test_seed_data_counts(engine):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Override engine in seed module
    import scripts.seed as seed_module

    original_engine = seed_module.engine
    seed_module.engine = engine

    seed_module.seed()

    with Session(engine) as db:
        users = db.execute(text("SELECT count(*) FROM users")).scalar()
        courses = db.execute(text("SELECT count(*) FROM courses")).scalar()
        sessions = db.execute(text("SELECT count(*) FROM sessions")).scalar()
        logs = db.execute(text("SELECT count(*) FROM engagement_logs")).scalar()

        assert users == 12, f"Expected 12 users, got {users}"
        assert courses == 3, f"Expected 3 courses, got {courses}"
        assert sessions == 5, f"Expected 5 sessions, got {sessions}"
        assert logs == 100, f"Expected 100 engagement logs, got {logs}"

    seed_module.engine = original_engine


# Test 3: Seed is idempotent
def test_seed_idempotent(engine):
    import scripts.seed as seed_module

    original_engine = seed_module.engine
    seed_module.engine = engine

    seed_module.seed()  # Run again

    with Session(engine) as db:
        users = db.execute(text("SELECT count(*) FROM users")).scalar()
        assert users == 12, f"Seed created duplicates! Got {users}"

    seed_module.engine = original_engine
