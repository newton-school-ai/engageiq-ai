"""Tests for Alembic migrations and seed data."""

import os

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/engageiq_dev")


@pytest.fixture(scope="module")
def engine():
    return create_engine(DATABASE_URL)


# Test 1: Migration created all required tables
def test_migration_creates_tables(engine):
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    required = ["users", "courses", "sessions", "engagement_logs"]
    for table in required:
        assert table in tables, f"Table '{table}' missing"


# Test 2: Seed data counts are correct
def test_seed_data_counts(engine):
    with Session(engine) as db:
        users = db.execute(text("SELECT count(*) FROM users")).scalar()
        courses = db.execute(text("SELECT count(*) FROM courses")).scalar()
        sessions = db.execute(text("SELECT count(*) FROM sessions")).scalar()
        logs = db.execute(text("SELECT count(*) FROM engagement_logs")).scalar()

        assert users == 12, f"Expected 12 users (2 teachers + 10 students), got {users}"
        assert courses == 3, f"Expected 3 courses, got {courses}"
        assert sessions == 5, f"Expected 5 sessions, got {sessions}"
        assert logs == 100, f"Expected 100 engagement logs, got {logs}"


# Test 3: Seed is idempotent (running twice does not duplicate)
def test_seed_idempotent(engine):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.seed import seed

    seed()  # Run again
    with Session(engine) as db:
        users = db.execute(text("SELECT count(*) FROM users")).scalar()
        assert users == 12, "Seed created duplicates!"
