"""Planner node — converts the user task into a URL + driver goal."""
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from browser_agent.state import BrowserState
from browser_agent.prompts.templates import PLANNER_SYSTEM, PLANNER_USER

load_dotenv()


class BrowserPlan(BaseModel):
    url: str = Field(description="The URL to visit")
    goal: str = Field(description="Specific goal for the browser driver")
    task_type: Literal["shopping", "informational"] = Field(description="Type of task")
    reason: str = Field(description="Brief explanation of the plan")


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
    llm = _get_llm().with_structured_output(BrowserPlan)

    messages = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=PLANNER_USER.format(task=state["task"])),
    ]

    try:
        plan: BrowserPlan = await llm.ainvoke(messages)
        print(f"[planner] url       : {plan.url}")
        print(f"[planner] goal      : {plan.goal}")
        print(f"[planner] task_type : {plan.task_type}")
        return {
            "browser_url": plan.url,
            "browser_goal": plan.goal,
            "task_type": plan.task_type,
            "status": "browsing",
        }
    except Exception as exc:
        print(f"[planner] error: {exc}")
        return {
            "browser_url": "",
            "browser_goal": state["task"],
            "task_type": "informational",
            "status": "browsing",
        }
