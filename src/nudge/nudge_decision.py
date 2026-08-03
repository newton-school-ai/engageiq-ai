"""Nudge decision logic: when to nudge based on state, duration, history."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.agents.nudge_agent import nudge_app


@dataclass
class NudgeDecision:
    """Determines whether a nudge should be sent and what kind."""

    should_nudge: bool
    nudge_type: Optional[str] = None
    reason: Optional[str] = None


class NudgeDecisionEngine:
    """Evaluates when and how to nudge using a LangGraph-based agent."""

    def __init__(self, cooldown_seconds: int = 300, max_nudges: int = 5):
        self.cooldown_seconds = cooldown_seconds
        self.max_nudges = max_nudges

    def should_nudge(
        self,
        current_state: str,
        state_duration: float,
        last_nudge_time: Optional[float],
        session_nudge_count: int,
        effectiveness_history: List[Dict],
    ) -> NudgeDecision:
        """Decide whether to nudge the student using LangGraph.

        Args:
            current_state: The current engagement state (e.g., 'distracted').
            state_duration: How long the user has been in the current state (seconds).
            last_nudge_time: Time elapsed since the last nudge (seconds), or None.
            session_nudge_count: Total nudges sent in the current session.
            effectiveness_history: List of past nudge effectiveness records.

        Returns:
            NudgeDecision containing should_nudge, nudge_type, and reason.
        """
        # Prepare the state for the LangGraph agent
        initial_state = {
            "current_state": current_state,
            "state_duration": state_duration,
            "last_nudge_time": last_nudge_time,
            "session_nudge_count": session_nudge_count,
            "effectiveness_history": effectiveness_history,
            "cooldown_seconds": self.cooldown_seconds,
            "max_nudges": self.max_nudges,
            # Defaults
            "should_nudge": False,
            "nudge_type": None,
            "reason": None,
        }

        # Invoke the LangGraph agent
        final_state = nudge_app.invoke(initial_state)

        return NudgeDecision(
            should_nudge=final_state.get("should_nudge", False),
            nudge_type=final_state.get("nudge_type"),
            reason=final_state.get("reason"),
        )
