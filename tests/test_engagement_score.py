"""Tests for multi-signal engagement scorer — Issue #16.

Covers:
    - All signals present (high, low, mixed)
    - Missing individual signals (weight redistribution)
    - All signals missing
    - Edge values (0 and 100)
    - Course type weight profiles
    - Score clamping
    - Signal contribution explainability helper
    - WeightProfile validation
"""

import pytest

from src.config.scoring_weights import PROFILES, WeightProfile, get_profile
from src.scoring.engagement_score import (
    compute_engagement_score,
    compute_engagement_score_from_profile,
    get_signal_contributions,
)


# ---------------------------------------------------------------------------
# 1. Fully engaged student
# ---------------------------------------------------------------------------


class TestFullyEngaged:
    def test_all_signals_max(self):
        score = compute_engagement_score(
            gaze_score=100,
            pose_score=100,
            expression_score=100,
            alertness_score=100,
        )
        assert score == pytest.approx(100.0)

    def test_all_signals_high_returns_above_90(self):
        score = compute_engagement_score(
            gaze_score=100,
            pose_score=95,
            expression_score=90,
            alertness_score=100,
        )
        assert score > 90.0


# ---------------------------------------------------------------------------
# 2. Fully disengaged student
# ---------------------------------------------------------------------------


class TestFullyDisengaged:
    def test_all_signals_zero(self):
        score = compute_engagement_score(
            gaze_score=0,
            pose_score=0,
            expression_score=0,
            alertness_score=0,
        )
        assert score == pytest.approx(0.0)

    def test_all_signals_low_returns_below_30(self):
        score = compute_engagement_score(
            gaze_score=10,
            pose_score=15,
            expression_score=5,
            alertness_score=20,
        )
        assert score < 30.0


# ---------------------------------------------------------------------------
# 3. Mixed signals
# ---------------------------------------------------------------------------


class TestMixedSignals:
    def test_looking_away_but_alert(self):
        """Gaze away (0) with good alertness and expression should score ~40-55."""
        score = compute_engagement_score(
            gaze_score=0,
            pose_score=30,
            expression_score=60,
            alertness_score=100,
        )
        assert 35.0 <= score <= 60.0

    def test_drowsy_student(self):
        """Alertness at zero with moderate other signals should score below 40."""
        score = compute_engagement_score(
            gaze_score=50,
            pose_score=40,
            expression_score=30,
            alertness_score=0,
        )
        assert score < 40.0

    def test_confused_but_attentive(self):
        """Low expression score but high gaze/pose/alertness — moderate range."""
        score = compute_engagement_score(
            gaze_score=90,
            pose_score=85,
            expression_score=20,
            alertness_score=95,
        )
        # With default weights: 0.3*90 + 0.2*85 + 0.25*20 + 0.25*95 = 72.75
        assert 65.0 <= score <= 80.0

    def test_weighted_average_matches_manual_calculation(self):
        """Verify score equals manual weighted sum using default weights."""
        g, p, e, a = 80.0, 70.0, 60.0, 90.0
        expected = 0.30 * g + 0.20 * p + 0.25 * e + 0.25 * a
        score = compute_engagement_score(
            gaze_score=g,
            pose_score=p,
            expression_score=e,
            alertness_score=a,
        )
        assert score == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# 4. Missing signals — weight redistribution
# ---------------------------------------------------------------------------


class TestMissingSignals:
    def test_missing_expression_redistributes_weight(self):
        """Missing expression: remaining weights scale up, score stays ~90."""
        score = compute_engagement_score(
            gaze_score=90,
            pose_score=85,
            expression_score=None,
            alertness_score=95,
        )
        # Remaining weights: gaze=0.30, pose=0.20, alertness=0.25 → total=0.75
        # Normalised: gaze=0.4, pose=0.267, alertness=0.333
        # Expected ≈ 0.4*90 + 0.267*85 + 0.333*95 ≈ 90.3
        assert score == pytest.approx(90.0, abs=3.0)

    def test_missing_gaze_still_scores_correctly(self):
        score_with_gaze = compute_engagement_score(
            gaze_score=80,
            pose_score=80,
            expression_score=80,
            alertness_score=80,
        )
        score_without_gaze = compute_engagement_score(
            gaze_score=None,
            pose_score=80,
            expression_score=80,
            alertness_score=80,
        )
        # All remaining signals equal → same output regardless of missing gaze
        assert score_without_gaze == pytest.approx(80.0, abs=1e-4)
        assert score_with_gaze == pytest.approx(80.0, abs=1e-4)

    def test_missing_alertness_does_not_crash(self):
        score = compute_engagement_score(
            gaze_score=70,
            pose_score=65,
            expression_score=75,
            alertness_score=None,
        )
        assert 0.0 <= score <= 100.0

    def test_only_one_signal_present(self):
        """Single signal should return its own value (weight=1.0 after normalisation)."""
        score = compute_engagement_score(
            gaze_score=72.0,
            pose_score=None,
            expression_score=None,
            alertness_score=None,
        )
        assert score == pytest.approx(72.0, abs=1e-4)

    def test_all_signals_none_returns_zero(self):
        score = compute_engagement_score(
            gaze_score=None,
            pose_score=None,
            expression_score=None,
            alertness_score=None,
        )
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. Edge values
# ---------------------------------------------------------------------------


class TestEdgeValues:
    def test_score_is_float(self):
        score = compute_engagement_score(
            gaze_score=80,
            pose_score=80,
            expression_score=80,
            alertness_score=80,
        )
        assert isinstance(score, float)

    def test_score_never_exceeds_100(self):
        score = compute_engagement_score(
            gaze_score=200,   # intentionally out-of-range input
            pose_score=200,
            expression_score=200,
            alertness_score=200,
        )
        assert score <= 100.0

    def test_score_never_below_zero(self):
        score = compute_engagement_score(
            gaze_score=-50,   # intentionally out-of-range input
            pose_score=-10,
            expression_score=-5,
            alertness_score=-20,
        )
        assert score >= 0.0

    def test_boundary_zero_and_hundred(self):
        assert compute_engagement_score(0, 0, 0, 0) == pytest.approx(0.0)
        assert compute_engagement_score(100, 100, 100, 100) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 6. Course type weight profiles
# ---------------------------------------------------------------------------


class TestCourseTypeProfiles:
    def test_theory_profile_weights_gaze_more(self):
        """Theory profile weights gaze at 0.35, higher than default 0.30."""
        assert PROFILES["theory"].gaze > PROFILES["default"].gaze

    def test_seminar_profile_weights_expression_most(self):
        """Seminar: expression is the dominant signal."""
        seminar = PROFILES["seminar"]
        assert seminar.expression >= seminar.gaze
        assert seminar.expression >= seminar.pose
        assert seminar.expression >= seminar.alertness

    def test_lab_profile_produces_different_score(self):
        """Same inputs, different course type → different score."""
        inputs = dict(
            gaze_score=80, pose_score=40, expression_score=90, alertness_score=70
        )
        default_score = compute_engagement_score(**inputs, course_type="default")
        lab_score = compute_engagement_score(**inputs, course_type="lab")
        assert default_score != lab_score

    def test_unknown_course_type_falls_back_to_default(self):
        score_unknown = compute_engagement_score(
            gaze_score=80,
            pose_score=80,
            expression_score=80,
            alertness_score=80,
            course_type="underwater_basket_weaving",
        )
        score_default = compute_engagement_score(
            gaze_score=80,
            pose_score=80,
            expression_score=80,
            alertness_score=80,
            course_type="default",
        )
        assert score_unknown == pytest.approx(score_default)

    def test_all_builtin_profiles_weights_sum_to_one(self):
        for name, profile in PROFILES.items():
            total = profile.gaze + profile.pose + profile.expression + profile.alertness
            assert total == pytest.approx(1.0, abs=1e-6), (
                f"Profile '{name}' weights sum to {total}, expected 1.0"
            )

    def test_compute_from_profile_name_convenience_wrapper(self):
        score_a = compute_engagement_score(
            gaze_score=75,
            pose_score=65,
            expression_score=85,
            alertness_score=80,
            course_type="seminar",
        )
        score_b = compute_engagement_score_from_profile(
            gaze_score=75,
            pose_score=65,
            expression_score=85,
            alertness_score=80,
            profile_name="seminar",
        )
        assert score_a == pytest.approx(score_b)


# ---------------------------------------------------------------------------
# 7. WeightProfile dataclass validation
# ---------------------------------------------------------------------------


class TestWeightProfile:
    def test_valid_profile_instantiates(self):
        p = WeightProfile(gaze=0.25, pose=0.25, expression=0.25, alertness=0.25)
        assert p.gaze == 0.25

    def test_invalid_profile_raises_value_error(self):
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            WeightProfile(gaze=0.50, pose=0.50, expression=0.50, alertness=0.50)

    def test_profile_is_immutable(self):
        p = WeightProfile(gaze=0.25, pose=0.25, expression=0.25, alertness=0.25)
        with pytest.raises((AttributeError, TypeError)):
            p.gaze = 0.99  # type: ignore[misc]

    def test_get_profile_returns_correct_type(self):
        p = get_profile("lab")
        assert isinstance(p, WeightProfile)

    def test_custom_profile_overrides_course_type(self):
        custom = WeightProfile(gaze=0.10, pose=0.10, expression=0.70, alertness=0.10)
        score = compute_engagement_score(
            gaze_score=0,
            pose_score=0,
            expression_score=100,
            alertness_score=0,
            profile=custom,
        )
        # expression dominates at 70% weight → score ≈ 70
        assert score == pytest.approx(70.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 8. Signal contribution explainability
# ---------------------------------------------------------------------------


class TestSignalContributions:
    def test_contributions_sum_to_total(self):
        result = get_signal_contributions(
            gaze_score=80,
            pose_score=70,
            expression_score=90,
            alertness_score=60,
        )
        present_sum = sum(
            v for k, v in result.items() if k != "total" and v is not None
        )
        assert present_sum == pytest.approx(result["total"], abs=0.1)

    def test_missing_signal_shows_none_in_contributions(self):
        result = get_signal_contributions(
            gaze_score=80,
            pose_score=None,
            expression_score=90,
            alertness_score=60,
        )
        assert result["pose"] is None
        assert result["gaze"] is not None

    def test_contributions_contain_all_keys(self):
        result = get_signal_contributions(80, 70, 90, 60)
        assert set(result.keys()) == {"gaze", "pose", "expression", "alertness", "total"}
