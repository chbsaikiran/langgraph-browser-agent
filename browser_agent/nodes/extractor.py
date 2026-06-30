import os
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from browser_agent.state import BrowserState
from browser_agent.browser.controller import BrowserController
from browser_agent.prompts.templates import EXTRACTOR_SYSTEM

load_dotenv()


class Product(BaseModel):
    name: str
    price: float | None = None
    brand: str | None = None
    processor: str | None = None
    ram: str | None = None
    storage: str | None = None
    display: str | None = None
    rating: float | None = None
    reviews: int | None = None


class ProductList(BaseModel):
    products: list[Product] = Field(default_factory=list)


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


async def _get_page_text(page) -> str:
    """
    Extract product-relevant text from the page.

    Strategy (in order):
      1. Amazon product card elements  — most precise, avoids nav/ads entirely
      2. Locate the "Results" keyword in body text and read from there
      3. Full body text with leading navigation stripped
    """

    # ── Strategy 1: target Amazon product card containers ────────────────────
    amazon_selectors = [
        "[data-component-type='s-search-result']",
        "[data-asin]:not([data-asin=''])",
    ]
    for selector in amazon_selectors:
        try:
            cards = await page.locator(selector).all()
            if len(cards) >= 2:
                texts: list[str] = []
                for card in cards[:8]:
                    t = (await card.inner_text()).strip()
                    # skip cards that are only a sponsored badge with no price
                    if len(t) > 80 and "₹" in t:
                        texts.append(t)
                if len(texts) >= 2:
                    return "\n\n=== PRODUCT ===\n\n".join(texts[:6])
        except Exception:
            continue

    # ── Strategy 2: find "Results" section in body text ──────────────────────
    try:
        body = await page.inner_text("body")
        for marker in ("Results\n", "results for", "Showing results"):
            idx = body.find(marker)
            if idx != -1:
                return body[idx: idx + 10_000]
    except Exception:
        pass

    # ── Strategy 3: body text, skip first 2 KB (navbar / filters) ────────────
    try:
        body = await page.inner_text("body")
        return body[2000:12_000]
    except Exception:
        return ""


async def extractor_node(state: BrowserState) -> dict:
    page = await BrowserController.get_page()
    page_text = await _get_page_text(page)

    if not page_text.strip():
        page_text = state.get("page_snapshot", "")[:8000]

    llm = _get_llm().with_structured_output(ProductList)
    messages = [
        SystemMessage(content=EXTRACTOR_SYSTEM),
        HumanMessage(content=f"Extract products from this page text:\n\n{page_text}"),
    ]

    try:
        result: ProductList = await llm.ainvoke(messages)
        new_items = [p.model_dump(exclude_none=True) for p in result.products[:5]]
    except Exception:
        new_items = []

    # Merge with previously extracted items, deduplicated by name
    existing = {item["name"]: item for item in state.get("extracted_items", [])}
    for item in new_items:
        existing.setdefault(item["name"], item)

    merged = list(existing.values())[:5]
    return {"extracted_items": merged}
