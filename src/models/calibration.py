"""Per-student calibration persistence model."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.models.base import Base


class Calibration(Base):
    """Calibration settings persisted per student.

    This model stores the computed personalization data used to calibrate
    detection thresholds (e.g., EAR for drowsiness and gaze thresholds derived
    from head-pose baselines), plus any baseline expression distribution
    captured during the calibration session.
    """

    __tablename__ = "calibrations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True
    )

    # EAR / drowsiness
    resting_ear: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ear_threshold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Head pose baseline
    # Pose returned by estimate_head_pose: (pitch, yaw, roll) in degrees.
    baseline_pose_pitch_deg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    baseline_pose_yaw_deg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Gaze thresholds:
    # We store the computed absolute offsets from the baseline so the
    # API can return a complete calibration payload.
    gaze_pitch_down_threshold_deg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    gaze_yaw_threshold_deg_left: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    gaze_yaw_threshold_deg_right: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )

    # Expression baseline distribution (e.g., probability histogram for
    # each expression class).
    baseline_expression_distribution: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )

    calibration_version: Mapped[str] = mapped_column(
        String(50), nullable=False, default="v1"
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationship
    user: Mapped["Any"] = relationship("User", back_populates="calibrations")

    def __repr__(self) -> str:
        return (
            f"<Calibration id={self.id} user_id={self.user_id} "
            f"ear_threshold={self.ear_threshold}>"
        )
