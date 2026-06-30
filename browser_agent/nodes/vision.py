from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from browser_agent.state import BrowserState
from browser_agent.browser.controller import BrowserController
from browser_agent.browser.snapshot import take_screenshot_b64
from browser_agent.prompts.templates import VISION_PROMPT

# llava:7b is available locally via Ollama — used as fallback when a11y tree is sparse
_vision_llm: ChatOllama | None = None


def _get_vision_llm() -> ChatOllama:
    global _vision_llm
    if _vision_llm is None:
        _vision_llm = ChatOllama(model="llava:7b", temperature=0)
    return _vision_llm


async def vision_node(state: BrowserState) -> dict:
    page = await BrowserController.get_page()
    b64 = await take_screenshot_b64(page)

    llm = _get_vision_llm()
    prompt = VISION_PROMPT.format(task=state["task"])

    message = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": prompt},
    ])

    try:
        response = await llm.ainvoke([message])
        vision_desc = response.content
    except Exception as exc:
        vision_desc = f"[Vision analysis unavailable: {exc}]"

    # Append llava's description to the existing a11y snapshot so the planner
    # has both structured roles and visual element names to work with.
    existing = state.get("page_snapshot", "")
    enhanced = f"{existing}\n\n[VISION: llava:7b]\n{vision_desc}".strip()

    return {
        "page_snapshot": enhanced,
        "use_vision": False,   # consumed — don't loop back into vision
    }
