"""Tests for the nudge decision agent."""

import pytest

from src.nudge.nudge_decision import NudgeDecisionEngine


@pytest.fixture
def engine():
    return NudgeDecisionEngine(cooldown_seconds=300, max_nudges=5)


def test_no_nudge_if_not_distracted(engine):
    """Should not nudge if user is attentive."""
    decision = engine.should_nudge(
        current_state="attentive",
        state_duration=60,
        last_nudge_time=None,
        session_nudge_count=0,
        effectiveness_history=[],
    )
    assert not decision.should_nudge
    assert decision.reason == "User is not distracted"


def test_no_nudge_if_duration_short(engine):
    """Should not nudge if user is distracted for less than 30 seconds."""
    decision = engine.should_nudge(
        current_state="distracted",
        state_duration=20,
        last_nudge_time=None,
        session_nudge_count=0,
        effectiveness_history=[],
    )
    assert not decision.should_nudge
    assert decision.reason == "Distraction duration too short"


def test_nudges_if_sustained_distraction(engine):
    """Should nudge if user is distracted for 30s+ and no cooldown applies."""
    decision = engine.should_nudge(
        current_state="distracted",
        state_duration=35,
        last_nudge_time=None,
        session_nudge_count=0,
        effectiveness_history=[],
    )
    assert decision.should_nudge
    assert decision.nudge_type == "gentle_reminder"


def test_no_nudge_during_cooldown(engine):
    """Should not nudge if within cooldown period."""
    decision = engine.should_nudge(
        current_state="distracted",
        state_duration=35,
        last_nudge_time=120,  # 120 seconds ago, < 300 cooldown
        session_nudge_count=1,
        effectiveness_history=[],
    )
    assert not decision.should_nudge
    assert decision.reason == "Nudge cooldown active"


def test_nudges_after_cooldown(engine):
    """Should nudge if cooldown has elapsed."""
    decision = engine.should_nudge(
        current_state="distracted",
        state_duration=35,
        last_nudge_time=350,  # 350 seconds ago, > 300 cooldown
        session_nudge_count=1,
        effectiveness_history=[],
    )
    assert decision.should_nudge
    assert decision.nudge_type == "gentle_reminder"


def test_no_nudge_if_max_reached(engine):
    """Should not nudge if max nudges for session reached."""
    decision = engine.should_nudge(
        current_state="distracted",
        state_duration=35,
        last_nudge_time=600,
        session_nudge_count=5,  # Max is 5
        effectiveness_history=[],
    )
    assert not decision.should_nudge
    assert decision.reason == "Maximum nudges per session reached"


def test_nudge_type_selection_based_on_history(engine):
    """Should select a different nudge type if history shows negative effectiveness."""
    decision = engine.should_nudge(
        current_state="distracted",
        state_duration=35,
        last_nudge_time=400,
        session_nudge_count=1,
        effectiveness_history=[{"timestamp": 100, "effectiveness": -0.2}],
    )
    assert decision.should_nudge
    assert decision.nudge_type == "strong_warning"
