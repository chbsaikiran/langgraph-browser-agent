import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from browser_agent.state import BrowserState
from browser_agent.prompts.templates import PLANNER_SYSTEM, PLANNER_USER

load_dotenv()


class BrowserAction(BaseModel):
    action: Literal["navigate", "click", "type", "key", "scroll", "set_price", "answer", "extract", "done"]
    target: str | None = Field(None, description="URL, element text, key name, or max price for set_price")
    value: str | None = Field(None, description="Text to type, scroll direction, or min price for set_price")
    reason: str = Field(description="Why this action is taken")


_llm: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0,
        )
    return _llm


async def planner_node(state: BrowserState) -> dict:
    llm = _get_llm().with_structured_output(BrowserAction)

    history = state.get("action_history", [])[-10:]
    history_text = "\n".join(
        f"  {i+1}. {h['action']} {h.get('target') or ''}"
        f"{' → ' + h.get('value', '') if h.get('value') else ''}"
        f" [{h.get('result', 'ok')}]"
        for i, h in enumerate(history)
    ) or "  (none)"

    current_url = state.get("current_url") or "(not yet navigated)"

    # Give the LLM a clear signal when the Amazon price filter is already applied
    if "rh=p_36" in current_url:
        url_hint = "NOTE: Price filter is already applied in the URL. If product results are visible in the snapshot, issue 'extract' NOW.\n"
    else:
        url_hint = ""

    prompt = PLANNER_USER.format(
        task=state["task"],
        url=current_url,
        url_hint=url_hint,
        snapshot=state.get("page_snapshot", "(empty)")[:4000],
        history=history_text,
    )

    messages = [SystemMessage(content=PLANNER_SYSTEM), HumanMessage(content=prompt)]

    try:
        action: BrowserAction = await llm.ainvoke(messages)
        return {
            "pending_action": action.model_dump(),
            "iteration": state.get("iteration", 0) + 1,
        }
    except Exception as exc:
        return {
            "pending_action": {"action": "done", "reason": f"Planner error: {exc}"},
            "error_count": state.get("error_count", 0) + 1,
            "iteration": state.get("iteration", 0) + 1,
        }
