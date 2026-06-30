"""LLM adapter that replaces S9's V9Client gateway.

A11y driver calls → Gemini (text, cheap)
Vision driver calls → llava:7b via Ollama (vision, local)
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()


@dataclass
class LLMResult:
    """Drop-in replacement for S9's GatewayResult."""
    parsed: dict | None
    text: str
    provider: str = "gemini"
    model: str = "gemini-2.0-flash"
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from text, handling markdown fences."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]+\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _content_to_str(content) -> str:
    if isinstance(content, list):
        return " ".join(
            p.get("text", str(p)) if isinstance(p, dict) else str(p)
            for p in content if p
        )
    return str(content)


class LLMClient:
    """Adapter presenting the same .chat() / .vision() interface as V9Client."""

    def __init__(self):
        self._gemini: ChatGoogleGenerativeAI | None = None
        self._llava: ChatOllama | None = None

    def _get_gemini(self) -> ChatGoogleGenerativeAI:
        if self._gemini is None:
            self._gemini = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0,
                model_kwargs={"response_mime_type": "application/json"},
            )
        return self._gemini

    def _get_llava(self) -> ChatOllama:
        if self._llava is None:
            self._llava = ChatOllama(model="llava:7b", temperature=0)
        return self._llava

    async def chat(
        self,
        prompt: str,
        *,
        system: str,
        schema: dict,
        schema_name: str = "",
        max_tokens: int = 1024,
        provider: str | None = None,
        model: str | None = None,
    ) -> LLMResult:
        t0 = time.time()
        schema_hint = (
            f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        )
        messages = [
            SystemMessage(content=system + schema_hint),
            HumanMessage(content=prompt),
        ]
        try:
            llm = self._get_gemini()
            resp = await llm.ainvoke(messages)
            text = _content_to_str(resp.content)
            parsed = _extract_json(text)
        except Exception as exc:
            text = str(exc)
            parsed = None
        return LLMResult(
            parsed=parsed,
            text=text,
            provider="gemini",
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            latency_ms=int((time.time() - t0) * 1000),
        )

    async def vision(
        self,
        image_data_url: str,
        prompt: str,
        *,
        system: str,
        schema: dict,
        schema_name: str = "",
        max_tokens: int = 1024,
        provider: str | None = None,
        model: str | None = None,
    ) -> LLMResult:
        t0 = time.time()
        schema_hint = (
            f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        )
        full_prompt = system + schema_hint + "\n\n" + prompt
        message = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text", "text": full_prompt},
        ])
        try:
            llm = self._get_llava()
            resp = await llm.ainvoke([message])
            text = _content_to_str(resp.content)
            parsed = _extract_json(text)
        except Exception as exc:
            text = str(exc)
            parsed = None
        return LLMResult(
            parsed=parsed,
            text=text,
            provider="ollama",
            model="llava:7b",
            latency_ms=int((time.time() - t0) * 1000),
        )
