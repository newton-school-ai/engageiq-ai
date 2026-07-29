"""Tests for calibration system (Issue #19)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.main import app
from src.database import get_db
from src.models.base import Base
from src.scoring.calibration import CalibrationManager

# Import models to register in SQLAlchemy metadata
from src.models.calibration import Calibration  # noqa: F401
from src.models.user import User  # noqa: F401

# In-memory SQLite DB for API + model tests
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db():
    """Create a fresh database for each test."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db):
    """Test client with overridden get_db dependency."""

    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _login(client: TestClient, credential: str) -> str:
    res = client.post("/api/v1/auth/token", json={"credential": credential})
    assert res.status_code == 200
    return res.json()["access_token"]


def test_compute_thresholds_ear_and_gaze_calibration(db):
    manager = CalibrationManager(
        db, ear_multiplier=0.8, calibration_duration_seconds=30.0
    )

    ear_readings = [0.30, 0.40, 0.35]  # median = 0.35 => threshold = 0.28
    pose_readings = [
        {"pitch": -5.0, "yaw": 10.0},
        {"pitch": -6.0, "yaw": 12.0},
        {"pitch": -4.0, "yaw": 9.0},
    ]
    baseline_dist = {"neutral": 0.7, "happy": 0.2, "sad": 0.1}

    computed = manager.compute_thresholds(
        ear_readings=ear_readings,
        pose_readings=pose_readings,
        baseline_expression_distribution=baseline_dist,
    )

    assert computed["resting_ear"] == pytest.approx(0.35)
    assert computed["ear_threshold"] == pytest.approx(0.35 * 0.8)

    # mean pitch = (-5-6-4)/3 = -5, mean yaw = (10+12+9)/3 = 10.333...
    assert computed["baseline_pose_pitch_deg"] == pytest.approx(-5.0)
    assert computed["baseline_pose_yaw_deg"] == pytest.approx(10.333333, rel=1e-6)

    # Gaze thresholds are baseline + defaults offsets (validated in manager)
    assert computed["gaze_yaw_threshold_deg_left"] < computed["baseline_pose_yaw_deg"]
    assert computed["gaze_yaw_threshold_deg_right"] > computed["baseline_pose_yaw_deg"]
    assert computed["baseline_expression_distribution"]["neutral"] == pytest.approx(0.7)


def test_recalibration_upserts_overwrite_values(db):
    manager = CalibrationManager(db, ear_multiplier=0.8)

    user_id = 1
    # Create a minimal user row using the same manager session
    from src.models.user import User

    db.add(User(id=user_id, email="mock_student@nst.edu", name="Mock", role="student"))  # type: ignore[arg-type]
    db.commit()

    computed1 = manager.compute_thresholds(
        ear_readings=[0.25, 0.25, 0.25],
        pose_readings=[{"pitch": 0.0, "yaw": 0.0}],
        baseline_expression_distribution={"neutral": 1.0},
    )
    calib1 = manager.upsert_calibration(user_id, computed1)
    assert calib1.ear_threshold == pytest.approx(0.25 * 0.8)

    computed2 = manager.compute_thresholds(
        ear_readings=[0.40, 0.40, 0.40],
        pose_readings=[{"pitch": 0.0, "yaw": 0.0}],
        baseline_expression_distribution={"neutral": 1.0},
    )
    calib2 = manager.upsert_calibration(user_id, computed2)
    assert calib2.id == calib1.id
    assert calib2.ear_threshold == pytest.approx(0.40 * 0.8)


def test_calibration_api_skip_and_get_default_behaviour(db, client):
    token = _login(client, "mock_student@nst.edu")

    # Fetch user_id from DB
    from src.models.user import User

    user = db.query(User).filter(User.email == "mock_student@nst.edu").first()
    assert user is not None

    user_id = user.id

    # Skip calibration
    res = client.post(
        f"/api/calibrate/{user_id}",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "session_duration_seconds": 10,
            "ear_readings": [0.2],
            "pose_readings": [{"pitch": 0.0, "yaw": 0.0}],
            "baseline_expression_distribution": {"neutral": 1.0},
            "skip_calibration": True,
        },
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["skipped"] is True
    assert payload["ear_threshold"] is None

    # Get calibration - should be skipped/default
    res2 = client.get(
        f"/api/calibrate/{user_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res2.status_code == 200
    payload2 = res2.json()
    assert payload2["skipped"] is True
    assert payload2["ear_threshold"] is None


def test_calibration_api_invalid_session_duration_rejected(db, client):
    token = _login(client, "mock_student2@nst.edu")

    from src.models.user import User

    user = db.query(User).filter(User.email == "mock_student2@nst.edu").first()
    assert user is not None
    user_id = user.id

    res = client.post(
        f"/api/calibrate/{user_id}",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "session_duration_seconds": 5.0,
            "ear_readings": [0.2, 0.21],
            "pose_readings": [{"pitch": 0.0, "yaw": 0.0}],
            "baseline_expression_distribution": {"neutral": 1.0},
            "skip_calibration": False,
        },
    )
    assert res.status_code == 400


def test_calibration_api_persists_and_gets_values(db, client):
    token = _login(client, "mock_student3@nst.edu")

    from src.models.user import User

    user = db.query(User).filter(User.email == "mock_student3@nst.edu").first()
    assert user is not None
    user_id = user.id

    res = client.post(
        f"/api/calibrate/{user_id}",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "session_duration_seconds": 30.0,
            "ear_readings": [0.30, 0.35, 0.40],  # median 0.35 => threshold 0.28
            "pose_readings": [
                {"pitch": -5.0, "yaw": 10.0},
                {"pitch": -6.0, "yaw": 12.0},
            ],
            "baseline_expression_distribution": {"neutral": 0.6, "happy": 0.4},
            "skip_calibration": False,
        },
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["skipped"] is False
    assert payload["ear_threshold"] is not None
    assert payload["resting_ear"] is not None

    res2 = client.get(
        f"/api/calibrate/{user_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res2.status_code == 200
    payload2 = res2.json()
    assert payload2["ear_threshold"] == payload["ear_threshold"]
    assert payload2["baseline_pose"] is not None
