"""Calibration API routes.

Implements per-student calibration that personalizes detection thresholds.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict

from sqlalchemy.orm import Session

from src.api.middleware.auth import get_current_user
from src.database import get_db
from src.scoring.calibration import CalibrationManager, PoseReading

logger = logging.getLogger(__name__)

router = APIRouter(tags=["calibration"])


class PoseReadingRequest(BaseModel):
    """Incoming head-pose sample used for calibration."""

    pitch: float
    yaw: float
    roll: Optional[float] = None

    model_config = ConfigDict(extra="forbid")


class CalibrationRequest(BaseModel):
    """Body for starting / completing a calibration session."""

    session_duration_seconds: float = Field(
        ..., ge=0.0, description="Client-reported duration of the calibration session."
    )

    ear_readings: list[float] = Field(
        ...,
        min_length=1,
        description="Captured EAR samples during the calibration session.",
    )

    pose_readings: list[PoseReadingRequest] = Field(
        ...,
        min_length=1,
        description="Captured head-pose samples (pitch/yaw/roll).",
    )

    baseline_expression_distribution: Dict[str, float] = Field(
        ..., description="Observed baseline expression distribution during calibration."
    )

    skip_calibration: bool = Field(
        default=False,
        description="If true, calibration is skipped and defaults will be used.",
    )

    model_config = ConfigDict(extra="forbid")


class CalibrationResponse(BaseModel):
    """Response payload for calibration retrieval/updates."""

    user_id: int
    calibration_version: str
    resting_ear: float | None
    ear_threshold: float | None

    baseline_pose: Dict[str, float] | None
    gaze_thresholds: Dict[str, float] | None
    baseline_expression_distribution: Dict[str, Any] | None

    created_at: str | None
    updated_at: str | None

    skipped: bool = False
    warning: str | None = None

    model_config = ConfigDict(extra="forbid")


@router.post(
    "/calibrate/{user_id}",
    response_model=CalibrationResponse,
    status_code=status.HTTP_200_OK,
)
def calibrate_user(
    user_id: int,
    payload: CalibrationRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CalibrationResponse:
    """Capture calibration session and persist calibration data for a student.

    Notes:
    - This endpoint requires the authenticated user to match {user_id}.
    - A 30-second session is required unless skip_calibration=true.
    """
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Cannot calibrate other users")

    manager = CalibrationManager(db)

    if payload.skip_calibration:
        logger.warning(
            "Calibration skipped for user_id=%s; defaults will be used.",
            user_id,
        )
        return CalibrationResponse(
            user_id=user_id,
            calibration_version="default",
            resting_ear=None,
            ear_threshold=None,
            baseline_pose=None,
            gaze_thresholds=None,
            baseline_expression_distribution=None,
            created_at=None,
            updated_at=None,
            skipped=True,
            warning="Calibration skipped; using default thresholds.",
        )

    if payload.session_duration_seconds < manager.calibration_duration_seconds:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Calibration session must be at least {manager.calibration_duration_seconds} seconds. "
                f"Got {payload.session_duration_seconds}."
            ),
        )

    # Compute calibration
    pose_readings: list[PoseReading] = [
        {"pitch": pr.pitch, "yaw": pr.yaw, "roll": pr.roll} for pr in payload.pose_readings
    ]

    computed = manager.compute_thresholds(
        ear_readings=payload.ear_readings,
        pose_readings=pose_readings,
        baseline_expression_distribution=payload.baseline_expression_distribution,
    )

    calib = manager.upsert_calibration(
        user_id,
        computed,
        baseline_expression_distribution=computed["baseline_expression_distribution"],
    )

    data = manager.serialize(calib)
    return CalibrationResponse(
        user_id=data["user_id"],
        calibration_version=data["calibration_version"],
        resting_ear=data["resting_ear"],
        ear_threshold=data["ear_threshold"],
        baseline_pose=data["baseline_pose"],
        gaze_thresholds=data["gaze_thresholds"],
        baseline_expression_distribution=data["baseline_expression_distribution"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        skipped=False,
        warning=None,
    )


@router.get(
    "/calibrate/{user_id}",
    response_model=CalibrationResponse,
    status_code=status.HTTP_200_OK,
)
def get_user_calibration(
    user_id: int,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CalibrationResponse:
    """Get persisted calibration for a student."""
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Cannot access other users")

    manager = CalibrationManager(db)
    calib = manager.get_calibration(user_id)

    if calib is None:
        return CalibrationResponse(
            user_id=user_id,
            calibration_version="default",
            resting_ear=None,
            ear_threshold=None,
            baseline_pose=None,
            gaze_thresholds=None,
            baseline_expression_distribution=None,
            created_at=None,
            updated_at=None,
            skipped=True,
            warning="No calibration found; using default thresholds.",
        )

    data = manager.serialize(calib)
    return CalibrationResponse(
        user_id=data["user_id"],
        calibration_version=data["calibration_version"],
        resting_ear=data["resting_ear"],
        ear_threshold=data["ear_threshold"],
        baseline_pose=data["baseline_pose"],
        gaze_thresholds=data["gaze_thresholds"],
        baseline_expression_distribution=data["baseline_expression_distribution"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        skipped=False,
        warning=None,
    )
