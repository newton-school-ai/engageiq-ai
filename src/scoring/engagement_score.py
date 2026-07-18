"""Multi-signal engagement scorer.

Fuses four CV signals — gaze, head pose, expression, and alertness — into a
single 0-100 engagement score using a configurable weighted average.

Design decisions:
- Pure function: compute_engagement_score has no side effects and no state.
  This makes it trivial to test, cache, or call from any context.
- Missing signal handling: when a CV model fails or is unavailable, its weight
  is redistributed proportionally among the remaining signals so the score
  stays in the 0-100 range without biasing toward lower values.
- Float output: returning a float (not int) enables fine-grained temporal
  smoothing in the downstream filter (#18).
- Configurable weights: accepts an optional WeightProfile so callers can
  switch between course types (theory, lab, seminar, discussion) without
  changing this function.
"""

from typing import Optional

from src.config.scoring_weights import PROFILES, WeightProfile, get_profile


def compute_engagement_score(
    gaze_score: Optional[float],
    pose_score: Optional[float],
    expression_score: Optional[float],
    alertness_score: Optional[float],
    course_type: str = "default",
    profile: Optional[WeightProfile] = None,
) -> float:
    """Compute a weighted engagement score from multiple CV signals.

    Each signal is expected in the 0-100 range:
        0   = fully disengaged / worst case for that signal
        100 = fully engaged / best case for that signal

    When a signal value is None (model failed or unavailable), its weight is
    redistributed proportionally among the signals that did produce a value.
    If all signals are None, returns 0.0.

    Args:
        gaze_score:       0-100 score from the gaze classifier.
                          100 = looking directly at screen.
        pose_score:       0-100 score from head pose estimator.
                          100 = head facing screen, no extreme tilt/turn.
        expression_score: 0-100 score from expression classifier.
                          100 = engaged/positive expression.
        alertness_score:  0-100 score from drowsiness + yawn detectors.
                          100 = fully alert, no drowsiness or yawning.
        course_type:      Weight profile name. One of "default", "theory",
                          "lab", "seminar", "discussion". Ignored when
                          `profile` is provided directly.
        profile:          Optional WeightProfile override. When provided,
                          `course_type` is ignored.

    Returns:
        Engagement score as a float in [0.0, 100.0].
        0.0  = fully disengaged.
        100.0 = fully engaged.
    """
    # Resolve the weight profile to use
    weights = profile if profile is not None else get_profile(course_type)

    # Map each signal to its configured weight
    signals = {
        "gaze": (gaze_score, weights.gaze),
        "pose": (pose_score, weights.pose),
        "expression": (expression_score, weights.expression),
        "alertness": (alertness_score, weights.alertness),
    }

    # Separate present signals from missing ones
    present = {
        name: (value, weight)
        for name, (value, weight) in signals.items()
        if value is not None
    }

    if not present:
        # All signals missing — cannot compute a meaningful score
        return 0.0

    # Total weight of signals that are present
    total_weight = sum(weight for _, weight in present.values())

    # Weighted average, normalised so weights always sum to 1.0
    score = sum(
        value * (weight / total_weight) for value, weight in present.values()
    )

    # Clamp to [0.0, 100.0] to guard against out-of-range inputs
    return float(max(0.0, min(100.0, score)))


def compute_engagement_score_from_profile(
    gaze_score: Optional[float],
    pose_score: Optional[float],
    expression_score: Optional[float],
    alertness_score: Optional[float],
    profile_name: str,
) -> float:
    """Convenience wrapper: compute score using a named weight profile.

    Equivalent to compute_engagement_score(..., course_type=profile_name).

    Args:
        gaze_score:       0-100 gaze signal.
        pose_score:       0-100 head pose signal.
        expression_score: 0-100 expression signal.
        alertness_score:  0-100 alertness signal.
        profile_name:     Weight profile name (e.g. "lab", "seminar").

    Returns:
        Engagement score float in [0.0, 100.0].
    """
    return compute_engagement_score(
        gaze_score=gaze_score,
        pose_score=pose_score,
        expression_score=expression_score,
        alertness_score=alertness_score,
        course_type=profile_name,
    )


def get_signal_contributions(
    gaze_score: Optional[float],
    pose_score: Optional[float],
    expression_score: Optional[float],
    alertness_score: Optional[float],
    course_type: str = "default",
    profile: Optional[WeightProfile] = None,
) -> dict:
    """Return per-signal weighted contributions for score explainability.

    Useful for answering "why did I get a low score at minute 14?" —
    shows exactly how much each signal contributed to the final score.

    Args:
        Same as compute_engagement_score.

    Returns:
        Dict with keys: "gaze", "pose", "expression", "alertness", "total".
        Each value is the weighted contribution of that signal (0-100 scale).
        Missing signals have a contribution of None.
    """
    weights = profile if profile is not None else get_profile(course_type)

    signals = {
        "gaze": (gaze_score, weights.gaze),
        "pose": (pose_score, weights.pose),
        "expression": (expression_score, weights.expression),
        "alertness": (alertness_score, weights.alertness),
    }

    present = {
        name: (value, weight)
        for name, (value, weight) in signals.items()
        if value is not None
    }

    total_weight = sum(w for _, w in present.values()) if present else 1.0

    contributions = {}
    for name, (value, weight) in signals.items():
        if value is None:
            contributions[name] = None
        else:
            contributions[name] = round(value * (weight / total_weight), 2)

    contributions["total"] = compute_engagement_score(
        gaze_score=gaze_score,
        pose_score=pose_score,
        expression_score=expression_score,
        alertness_score=alertness_score,
        course_type=course_type,
        profile=profile,
    )

    return contributions
