"""LangGraph agent for nudge decisions with learning feedback loop."""

from typing import TypedDict, Any, Optional
from langgraph.graph import StateGraph, START, END

class NudgeAgentState(TypedDict):
    current_state: str
    state_duration: float
    last_nudge_time: Optional[float]
    session_nudge_count: int
    effectiveness_history: list[dict]
    cooldown_seconds: int
    max_nudges: int
    
    should_nudge: bool
    nudge_type: Optional[str]
    reason: Optional[str]


def evaluate_engagement_state(state: NudgeAgentState) -> dict:
    """Evaluate if the user is disengaged long enough to warrant a nudge."""
    if state["current_state"] != "distracted":
        return {"should_nudge": False, "reason": "User is not distracted"}
    
    # 30 seconds default threshold
    if state["state_duration"] < 30:
        return {"should_nudge": False, "reason": "Distraction duration too short"}
        
    return {"should_nudge": True, "reason": "Sustained distraction detected"}


def check_history(state: NudgeAgentState) -> dict:
    """Check cooldown and max nudges limits."""
    # If a previous node already decided not to nudge, we respect that
    if not state.get("should_nudge", True):
        return {}

    if state["session_nudge_count"] >= state["max_nudges"]:
        return {"should_nudge": False, "reason": "Maximum nudges per session reached"}
        
    if state["last_nudge_time"] is not None:
        if state["last_nudge_time"] < state["cooldown_seconds"]:
            return {"should_nudge": False, "reason": "Nudge cooldown active"}
            
    return {}


def select_nudge_type(state: NudgeAgentState) -> dict:
    """Select the type of nudge based on effectiveness history."""
    if not state.get("should_nudge", True):
        return {}
        
    # Basic logic: if we have history, try to use it. Otherwise, default.
    # We could make this more complex by actually analyzing effectiveness_history.
    # For now, we will return 'gentle_reminder' unless history says otherwise.
    # In a full implementation, we'd pick based on which type increased engagement the most.
    
    nudge_type = "gentle_reminder"
    
    if state.get("effectiveness_history"):
        # Just a dummy heuristic for now: 
        # If the last nudge wasn't effective, try a stronger one.
        last_nudge = state["effectiveness_history"][-1]
        if last_nudge.get("effectiveness", 0) < 0:
            nudge_type = "strong_warning"
            
    return {"nudge_type": nudge_type}


def create_nudge_graph() -> Any:
    workflow = StateGraph(NudgeAgentState)
    
    workflow.add_node("evaluate_engagement", evaluate_engagement_state)
    workflow.add_node("check_history", check_history)
    workflow.add_node("select_nudge_type", select_nudge_type)
    
    workflow.add_edge(START, "evaluate_engagement")
    workflow.add_edge("evaluate_engagement", "check_history")
    workflow.add_edge("check_history", "select_nudge_type")
    workflow.add_edge("select_nudge_type", END)
    
    return workflow.compile()

nudge_app = create_nudge_graph()
