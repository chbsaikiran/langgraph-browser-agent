"""Extractor node — pulls structured product data from browser_content in state."""
import os
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from browser_agent.state import BrowserState
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


async def extractor_node(state: BrowserState) -> dict:
    page_text = state.get("browser_content", "")

    if not page_text.strip():
        return {"extracted_items": []}

    llm = _get_llm().with_structured_output(ProductList)
    messages = [
        SystemMessage(content=EXTRACTOR_SYSTEM),
        HumanMessage(content=f"Extract products from this page text:\n\n{page_text[:12000]}"),
    ]

    try:
        result: ProductList = await llm.ainvoke(messages)
        items = [p.model_dump(exclude_none=True) for p in result.products[:5]]
    except Exception:
        items = []

    return {"extracted_items": items}
