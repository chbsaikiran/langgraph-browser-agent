"""Answer node — reads browser_content from state and answers the user's question."""
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from browser_agent.state import BrowserState
from browser_agent.prompts.templates import ANSWER_SYSTEM

load_dotenv()

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


async def answer_node(state: BrowserState) -> dict:
    page_text = state.get("browser_content", "")
    browser_path = state.get("browser_path", "")

    if browser_path == "blocked" or not page_text.strip():
        return {
            "final_answer": (
                "Could not retrieve page content. "
                "The page may be blocked (CAPTCHA/login wall) or returned no text."
            ),
            "status": "done",
        }

    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(
            content=(
                f"User task: {state['task']}\n\n"
                f"Page URL: {state.get('browser_url', '')}\n\n"
                f"Page content:\n{page_text[:20000]}"
            )
        ),
    ]

    try:
        response = await _get_llm().ainvoke(messages)
        content = response.content
        if isinstance(content, list):
            answer = "\n".join(
                p.get("text", str(p)) if isinstance(p, dict) else str(p)
                for p in content if p
            )
        else:
            answer = str(content)
    except Exception as exc:
        answer = f"Failed to generate answer: {exc}"

    return {"final_answer": answer, "status": "done"}
