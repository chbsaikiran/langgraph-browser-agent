from langgraph.graph import StateGraph, END

from browser_agent.state import BrowserState
from browser_agent.nodes.planner import planner_node
from browser_agent.nodes.executor import executor_node
from browser_agent.nodes.vision import vision_node
from browser_agent.nodes.extractor import extractor_node
from browser_agent.nodes.comparator import comparator_node
from browser_agent.nodes.responder import responder_node
from browser_agent.nodes.answer import answer_node

MAX_ITERATIONS = 30


# ── Routing functions ──────────────────────────────────────────────────────────

def route_after_planner(state: BrowserState) -> str:
    """
    Terminal signals skip the browser entirely:
      answer  → read current page and answer the question (info queries)
      extract → pull structured product data (shopping queries)
      done    → bail out with whatever we have
    Everything else goes to the executor for a browser action.
    """
    action = state.get("pending_action") or {}
    act = action.get("action", "")
    if act == "answer":
        return "answer"
    if act == "extract":
        return "extractor"
    if act == "done":
        return "responder"
    return "executor"


def route_after_executor(state: BrowserState) -> str:
    """After a browser action: vision fallback, or loop back to planner."""
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "extractor"
    if state.get("use_vision", False):
        return "vision"
    return "planner"


def route_after_extractor(state: BrowserState) -> str:
    """≥3 products → compare; otherwise keep browsing."""
    items = state.get("extracted_items", [])
    iteration = state.get("iteration", 0)
    if len(items) >= 3 or iteration >= MAX_ITERATIONS:
        return "comparator"
    # Safety: if extractor ran multiple times with no results, give up and compare
    if iteration > 8 and not items:
        return "comparator"
    return "planner"


# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(BrowserState)

    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("vision", vision_node)
    builder.add_node("answer", answer_node)       # informational queries
    builder.add_node("extractor", extractor_node) # product extraction
    builder.add_node("comparator", comparator_node)
    builder.add_node("responder", responder_node)

    builder.set_entry_point("planner")

    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "executor":  "executor",
            "answer":    "answer",
            "extractor": "extractor",
            "responder": "responder",
        },
    )

    builder.add_conditional_edges(
        "executor",
        route_after_executor,
        {"planner": "planner", "vision": "vision", "extractor": "extractor"},
    )

    # vision enriches the snapshot then hands back to planner
    builder.add_edge("vision", "planner")

    builder.add_conditional_edges(
        "extractor",
        route_after_extractor,
        {"comparator": "comparator", "planner": "planner"},
    )

    builder.add_edge("comparator", "responder")
    builder.add_edge("responder", END)

    # answer node produces final_answer directly — no further processing needed
    builder.add_edge("answer", END)

    return builder.compile()
