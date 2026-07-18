"""Scoring weight profiles per course type.

Each profile defines how much each CV signal contributes to the final
engagement score. Weights must sum to 1.0.

Profiles:
    default   — balanced weights, suitable for most lectures
    theory    — higher gaze/alertness; passive listening dominates
    lab       — higher pose weight; students look down at keyboards more
    seminar   — higher expression weight; participation and reaction matter
    discussion — expression and pose prioritized; gaze less critical
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class WeightProfile:
    """Immutable set of weights for the four engagement signals.

    All four weights must sum to 1.0.
    """

    gaze: float
    pose: float
    expression: float
    alertness: float

    def __post_init__(self) -> None:
        total = self.gaze + self.pose + self.expression + self.alertness
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.4f}. "
                f"(gaze={self.gaze}, pose={self.pose}, "
                f"expression={self.expression}, alertness={self.alertness})"
            )


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

PROFILES: Dict[str, WeightProfile] = {
    # Default: balanced across all four signals
    "default": WeightProfile(
        gaze=0.30,
        pose=0.20,
        expression=0.25,
        alertness=0.25,
    ),
    # Theory lecture: student mostly watches the screen; gaze and alertness dominate
    "theory": WeightProfile(
        gaze=0.35,
        pose=0.15,
        expression=0.20,
        alertness=0.30,
    ),
    # Lab / coding session: students look at keyboard/screen alternately;
    # head pose matters less, alertness and expression more
    "lab": WeightProfile(
        gaze=0.25,
        pose=0.15,
        expression=0.30,
        alertness=0.30,
    ),
    # Seminar: active verbal participation; expression is the primary signal
    "seminar": WeightProfile(
        gaze=0.20,
        pose=0.20,
        expression=0.40,
        alertness=0.20,
    ),
    # Discussion: reactions and body language dominate; gaze less critical
    "discussion": WeightProfile(
        gaze=0.15,
        pose=0.30,
        expression=0.35,
        alertness=0.20,
    ),
}


def get_profile(course_type: str) -> WeightProfile:
    """Return the weight profile for a given course type.

    Falls back to the "default" profile if the requested type is unknown.

    Args:
        course_type: One of "default", "theory", "lab", "seminar", "discussion".

    Returns:
        WeightProfile for the requested course type.
    """
    return PROFILES.get(course_type.lower(), PROFILES["default"])
