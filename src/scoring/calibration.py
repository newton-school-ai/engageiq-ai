"""Calibration system for per-student detection threshold personalization."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Dict, Iterable, Optional, TypedDict

from sqlalchemy.orm import Session

from src.config.settings import settings
from src.models.calibration import Calibration

logger = logging.getLogger(__name__)


class PoseReading(TypedDict, total=False):
    """Head pose reading used during calibration."""

    pitch: float
    yaw: float
    roll: Optional[float]


class CalibrationComputationResult(TypedDict):
    """Computed calibration thresholds and persisted fields."""

    resting_ear: float
    ear_threshold: float
    baseline_pose_pitch_deg: float
    baseline_pose_yaw_deg: float
    gaze_pitch_down_threshold_deg: float
    gaze_yaw_threshold_deg_left: float
    gaze_yaw_threshold_deg_right: float
    baseline_expression_distribution: Dict[str, float]


@dataclass(frozen=True)
class CalibrationState:
    """Represents the state of a calibration operation."""

    user_id: int
    started_at: float
    ends_at: float

    resting_ear: Optional[float] = None
    baseline_pitch_deg: Optional[float] = None
    baseline_yaw_deg: Optional[float] = None
    baseline_expression_distribution: Optional[Dict[str, float]] = None

    @property
    def duration_seconds(self) -> float:
        return self.ends_at - self.started_at


class CalibrationManager:
    """Compute, persist, and retrieve student calibration data."""

    def __init__(
        self,
        db: Session,
        *,
        ear_multiplier: float = 0.8,
        calibration_duration_seconds: float = 30.0,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.db = db
        self.ear_multiplier = ear_multiplier
        self.calibration_duration_seconds = calibration_duration_seconds
        self.logger = logger_ or logger

    def get_calibration(self, user_id: int) -> Optional[Calibration]:
        """Fetch persisted calibration for a student."""
        return (
            self.db.query(Calibration)
            .filter(Calibration.user_id == user_id)
            .one_or_none()
        )

    def upsert_calibration(
        self,
        user_id: int,
        computed: CalibrationComputationResult,
        *,
        calibration_version: str = "v1",
        baseline_expression_distribution: Optional[Dict[str, float]] = None,
    ) -> Calibration:
        """Insert or update calibration payload for a student."""
        existing = self.get_calibration(user_id)
        payload = baseline_expression_distribution or computed[
            "baseline_expression_distribution"
        ]

        if existing is None:
            calib = Calibration(
                user_id=user_id,
                resting_ear=computed["resting_ear"],
                ear_threshold=computed["ear_threshold"],
                baseline_pose_pitch_deg=computed["baseline_pose_pitch_deg"],
                baseline_pose_yaw_deg=computed["baseline_pose_yaw_deg"],
                gaze_pitch_down_threshold_deg=computed["gaze_pitch_down_threshold_deg"],
                gaze_yaw_threshold_deg_left=computed["gaze_yaw_threshold_deg_left"],
                gaze_yaw_threshold_deg_right=computed["gaze_yaw_threshold_deg_right"],
                baseline_expression_distribution=payload,
                calibration_version=calibration_version,
            )
            self.db.add(calib)
            self.db.commit()
            self.db.refresh(calib)
            return calib

        # Recalibration: overwrite persisted values
        existing.resting_ear = computed["resting_ear"]
        existing.ear_threshold = computed["ear_threshold"]
        existing.baseline_pose_pitch_deg = computed["baseline_pose_pitch_deg"]
        existing.baseline_pose_yaw_deg = computed["baseline_pose_yaw_deg"]
        existing.gaze_pitch_down_threshold_deg = computed["gaze_pitch_down_threshold_deg"]
        existing.gaze_yaw_threshold_deg_left = computed["gaze_yaw_threshold_deg_left"]
        existing.gaze_yaw_threshold_deg_right = computed["gaze_yaw_threshold_deg_right"]
        existing.baseline_expression_distribution = payload
        existing.calibration_version = calibration_version
        self.db.commit()
        self.db.refresh(existing)
        return existing

    def compute_thresholds(
        self,
        *,
        ear_readings: Iterable[float],
        pose_readings: Iterable[PoseReading],
        baseline_expression_distribution: Dict[str, float],
    ) -> CalibrationComputationResult:
        """Compute calibration thresholds from captured session data."""
        ear_list = [float(x) for x in ear_readings if x is not None]
        if not ear_list:
            raise ValueError("ear_readings must contain at least one value")

        pitch_list = []
        yaw_list = []
        for pr in pose_readings:
            if pr.get("pitch") is not None:
                pitch_list.append(float(pr["pitch"]))
            if pr.get("yaw") is not None:
                yaw_list.append(float(pr["yaw"]))

        if not pitch_list or not yaw_list:
            raise ValueError("pose_readings must contain at least one pitch and yaw")

        # Resting EAR: use median for robustness to occasional blinks.
        resting_ear = float(median(ear_list))
        ear_threshold = float(resting_ear * self.ear_multiplier)

        baseline_pose_pitch_deg = float(mean(pitch_list))
        baseline_pose_yaw_deg = float(mean(yaw_list))

        # Gaze thresholds adjusted relative to natural head pose baseline.
        # settings values are treated as offsets from the baseline neutral pose.
        gaze_pitch_down_threshold_deg = float(
            baseline_pose_pitch_deg + settings.gaze_pitch_down_threshold_deg
        )
        gaze_yaw_threshold_deg_left = float(
            baseline_pose_yaw_deg - settings.gaze_yaw_threshold_deg
        )
        gaze_yaw_threshold_deg_right = float(
            baseline_pose_yaw_deg + settings.gaze_yaw_threshold_deg
        )

        if not baseline_expression_distribution:
            raise ValueError("baseline_expression_distribution must not be empty")

        # Ensure values are floats.
        baseline_dist = {k: float(v) for k, v in baseline_expression_distribution.items()}

        return CalibrationComputationResult(
            resting_ear=resting_ear,
            ear_threshold=ear_threshold,
            baseline_pose_pitch_deg=baseline_pose_pitch_deg,
            baseline_pose_yaw_deg=baseline_pose_yaw_deg,
            gaze_pitch_down_threshold_deg=gaze_pitch_down_threshold_deg,
            gaze_yaw_threshold_deg_left=gaze_yaw_threshold_deg_left,
            gaze_yaw_threshold_deg_right=gaze_yaw_threshold_deg_right,
            baseline_expression_distribution=baseline_dist,
        )

    def serialize(self, calibration: Calibration) -> Dict[str, Any]:
        """Convert a Calibration model into an API-friendly dict."""
        return {
            "user_id": calibration.user_id,
            "calibration_version": calibration.calibration_version,
            "resting_ear": calibration.resting_ear,
            "ear_threshold": calibration.ear_threshold,
            "baseline_pose": {
                "pitch_deg": calibration.baseline_pose_pitch_deg,
                "yaw_deg": calibration.baseline_pose_yaw_deg,
            },
            "gaze_thresholds": {
                "pitch_down_deg": calibration.gaze_pitch_down_threshold_deg,
                "yaw_left_deg": calibration.gaze_yaw_threshold_deg_left,
                "yaw_right_deg": calibration.gaze_yaw_threshold_deg_right,
            },
            "baseline_expression_distribution": calibration.baseline_expression_distribution,
            "created_at": calibration.created_at.isoformat() if calibration.created_at else None,
            "updated_at": calibration.updated_at.isoformat() if calibration.updated_at else None,
        }

    def apply_or_default_gaze_thresholds(self, calibration: Optional[Calibration]) -> Dict[str, float]:
        """Return threshold values to drive gaze classification."""
        if calibration is None:
            # Defaults are relative to a neutral (0) pose.
            return {
                "ear_closed_threshold": settings.gaze_ear_closed_threshold,
                "pitch_down_threshold_deg": settings.gaze_pitch_down_threshold_deg,
                "yaw_left_threshold_deg": -settings.gaze_yaw_threshold_deg,
                "yaw_right_threshold_deg": settings.gaze_yaw_threshold_deg,
            }

        return {
            "ear_closed_threshold": settings.gaze_ear_closed_threshold,  # separate gating; keep default for now
            "pitch_down_threshold_deg": calibration.gaze_pitch_down_threshold_deg or settings.gaze_pitch_down_threshold_deg,
            "yaw_left_threshold_deg": calibration.gaze_yaw_threshold_deg_left
            if calibration.gaze_yaw_threshold_deg_left is not None
            else -settings.gaze_yaw_threshold_deg,
            "yaw_right_threshold_deg": calibration.gaze_yaw_threshold_deg_right
            if calibration.gaze_yaw_threshold_deg_right is not None
            else settings.gaze_yaw_threshold_deg,
        }

    def start_calibration_state(self, user_id: int) -> CalibrationState:
        """Create an in-memory calibration state (client is responsible for capture)."""
        now = time.time()
        return CalibrationState(
            user_id=user_id,
            started_at=now,
            ends_at=now + self.calibration_duration_seconds,
        )
