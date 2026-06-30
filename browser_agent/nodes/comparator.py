import json
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from browser_agent.state import BrowserState
from browser_agent.prompts.templates import COMPARATOR_SYSTEM

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


async def comparator_node(state: BrowserState) -> dict:
    items = state.get("extracted_items", [])[:3]

    if not items:
        return {"comparison_result": "No products were extracted — nothing to compare."}

    products_json = json.dumps(items, indent=2, ensure_ascii=False)
    messages = [
        SystemMessage(content=COMPARATOR_SYSTEM),
        HumanMessage(content=f"Compare these products:\n\n{products_json}"),
    ]

    try:
        response = await _get_llm().ainvoke(messages)
        content = response.content
        if isinstance(content, list):
            content = "\n".join(
                p.get("text", str(p)) if isinstance(p, dict) else str(p)
                for p in content if p
            )
        return {"comparison_result": str(content)}
    except Exception as exc:
        return {"comparison_result": f"Comparison failed: {exc}"}
