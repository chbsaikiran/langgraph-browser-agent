from langgraph.graph import StateGraph, END

from browser_agent.state import BrowserState
from browser_agent.nodes.planner import planner_node
from browser_agent.nodes.browser_skill import browser_skill_node
from browser_agent.nodes.answer import answer_node
from browser_agent.nodes.extractor import extractor_node
from browser_agent.nodes.comparator import comparator_node
from browser_agent.nodes.responder import responder_node


def route_after_browser(state: BrowserState) -> str:
    """Route based on task type and whether the cascade succeeded."""
    path = state.get("browser_path", "")
    task_type = state.get("task_type", "informational")

    if path == "blocked":
        return "responder"
    if task_type == "shopping":
        return "extractor"
    return "answer"


def build_graph():
    builder = StateGraph(BrowserState)

    builder.add_node("planner",       planner_node)
    builder.add_node("browser_skill", browser_skill_node)
    builder.add_node("answer",        answer_node)
    builder.add_node("extractor",     extractor_node)
    builder.add_node("comparator",    comparator_node)
    builder.add_node("responder",     responder_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "browser_skill")

    builder.add_conditional_edges(
        "browser_skill",
        route_after_browser,
        {
            "answer":    "answer",
            "extractor": "extractor",
            "responder": "responder",
        },
    )

    builder.add_edge("answer",     END)
    builder.add_edge("extractor",  "comparator")
    builder.add_edge("comparator", "responder")
    builder.add_edge("responder",  END)

    return builder.compile()
