import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from browser_agent.state import BrowserState
from browser_agent.browser.controller import BrowserController
from browser_agent.prompts.templates import ANSWER_SYSTEM

load_dotenv()

_llm: ChatGoogleGenerativeAI | None = None

# Selectors for main article/content area — tried in order before falling back to body
_CONTENT_SELECTORS = [
    "#mw-content-text",       # Wikipedia article body
    ".mw-body-content",       # Wikipedia (alt)
    "article",                # HTML5 semantic article
    "main",                   # HTML5 main content
    "[role='main']",          # ARIA landmark
    "#content",               # Common CMS id
    ".post-content",          # Blog posts
    ".article-body",          # News articles
    ".entry-content",         # WordPress
]


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0,
        )
    return _llm


async def _get_article_text(page) -> str:
    """
    Extract the main readable content from the page.
    Tries semantic/known selectors first, falls back to full body.
    """
    for selector in _CONTENT_SELECTORS:
        try:
            loc = page.locator(selector).first
            text = await loc.inner_text(timeout=3_000)
            if text and len(text.strip()) > 300:
                return text.strip()[:20_000]
        except Exception:
            continue

    # Fallback: body text, skip first 1 KB of navigation boilerplate
    try:
        body = await page.inner_text("body")
        return body[1000:21_000]
    except Exception:
        return ""


async def answer_node(state: BrowserState) -> dict:
    """
    Read the current page's content and answer the user's question using Gemini.
    Used for informational queries (Wikipedia lookups, article reading, fact extraction).
    """
    page = await BrowserController.get_page()
    page_text = await _get_article_text(page)

    if not page_text.strip():
        return {
            "final_answer": "Could not read page content. The page may be empty or blocked.",
            "status": "done",
        }

    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(
            content=(
                f"User task: {state['task']}\n\n"
                f"Page URL: {state.get('current_url', '')}\n\n"
                f"Page content:\n{page_text}"
            )
        ),
    ]

    try:
        response = await _get_llm().ainvoke(messages)
        # Gemini can return content as a list of blocks instead of a plain string
        content = response.content
        if isinstance(content, list):
            answer = "\n".join(
                part.get("text", str(part)) if isinstance(part, dict) else str(part)
                for part in content
                if part
            )
        else:
            answer = str(content)
    except Exception as exc:
        answer = f"Failed to generate answer: {exc}"

    return {"final_answer": answer, "status": "done"}
