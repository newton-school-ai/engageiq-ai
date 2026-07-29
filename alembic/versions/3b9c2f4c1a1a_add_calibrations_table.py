"""add calibrations table

Revision ID: 3b9c2f4c1a1a
Revises: e55261113c57
Create Date: 2026-07-29 00:00:00

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3b9c2f4c1a1a"
down_revision: Union[str, Sequence[str], None] = "e55261113c57"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "calibrations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("resting_ear", sa.Float(), nullable=True),
        sa.Column("ear_threshold", sa.Float(), nullable=True),
        sa.Column("baseline_pose_pitch_deg", sa.Float(), nullable=True),
        sa.Column("baseline_pose_yaw_deg", sa.Float(), nullable=True),
        sa.Column("gaze_pitch_down_threshold_deg", sa.Float(), nullable=True),
        sa.Column("gaze_yaw_threshold_deg_left", sa.Float(), nullable=True),
        sa.Column("gaze_yaw_threshold_deg_right", sa.Float(), nullable=True),
        sa.Column("baseline_expression_distribution", sa.JSON(), nullable=True),
        sa.Column("calibration_version", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("calibrations")
