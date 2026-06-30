"""4-layer browser cascade adapted from S9SharedCode BrowserSkill.

Layer 1  — HTML extract via trafilatura (no LLM, no browser)
Layer 2b — A11yDriver        (Gemini text, Playwright)
Layer 3  — SetOfMarksDriver  (llava:7b vision, Playwright)

Gateway-access blocks (CAPTCHA / Cloudflare / login walls) detected after
JS render are surfaced as path="blocked" so callers can handle them.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import trafilatura
from playwright.async_api import async_playwright

from .llm_client import LLMClient
from .driver import A11yDriver, DriverConfig, DriverResult, SetOfMarksDriver


# ── gateway-block detection ──────────────────────────────────────────────────
_GATEWAY_BLOCK_MARKERS = (
    ("captcha",    "Let's confirm you are human"),
    ("captcha",    "Enter the characters you see below"),
    ("captcha",    "Robot Check"),
    ("captcha",    "Please verify you are a human"),
    ("captcha",    "/errors/validateCaptcha"),
    ("hcaptcha",   'class="h-captcha"'),
    ("hcaptcha",   "data-hcaptcha-widget-id"),
    ("recaptcha",  'class="g-recaptcha"'),
    ("recaptcha",  "g-recaptcha-response"),
    ("cloudflare", "Checking your browser before accessing"),
    ("cloudflare", "cf-browser-verification"),
    ("cloudflare", "cf-challenge-running"),
    ("login_wall", "You must be logged in"),
    ("login_wall", "Sign in to continue"),
    ("login_wall", "Please log in to continue"),
)


def detect_gateway_block(html: str) -> str | None:
    if not html:
        return None
    h = html.lower()
    for kind, needle in _GATEWAY_BLOCK_MARKERS:
        if needle.lower() in h:
            return kind
    return None


# ── Layer 1 helpers ──────────────────────────────────────────────────────────
_UA = "Mozilla/5.0 (compatible; BrowserAgent/1.0)"


async def _fetch_html(url: str, timeout: float = 30.0) -> tuple[str, str]:
    """Return (html, final_url)."""
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True,
        headers={"User-Agent": _UA},
    ) as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.text, str(r.url)


def _trafilatura_extract(html: str) -> str:
    text = trafilatura.extract(
        html, include_links=True, include_formatting=False, favor_recall=True,
    )
    return (text or "").strip()


def _is_useful_extract(content: str, goal: str) -> bool:
    """Layer 1 is enough when content is long AND goal doesn't require interaction."""
    if len(content) < 200:
        return False
    interactive_verbs = ("click", "fill", "select", "type", "drag",
                         "filter", "sort", "submit", "navigate", "search")
    if any(v in goal.lower() for v in interactive_verbs):
        return False
    return True


# ── Result type ──────────────────────────────────────────────────────────────
@dataclass
class CascadeResult:
    success: bool
    path: str           # "extract" | "a11y" | "vision" | "blocked" | "error"
    content: str        # extracted text for downstream nodes
    final_url: str
    actions: list[dict] = field(default_factory=list)
    error: str | None = None


# ── Browser cascade ──────────────────────────────────────────────────────────
class BrowserCascade:
    def __init__(
        self,
        *,
        max_steps_a11y: int = 15,
        max_steps_vision: int = 12,
        artifacts_dir: Optional[str] = None,
    ):
        self.max_steps_a11y = max_steps_a11y
        self.max_steps_vision = max_steps_vision
        self.artifacts_dir = artifacts_dir
        self._client = LLMClient()

    async def run(self, url: str, goal: str) -> CascadeResult:
        print(f"[cascade] url  : {url}")
        print(f"[cascade] goal : {goal}")

        # ── Layer 1: httpx + trafilatura ────────────────────────────────────
        print("[cascade] layer1: fetching HTML via httpx ...")
        layer1_error: str | None = None
        html, final_url = "", url
        try:
            html, final_url = await _fetch_html(url)
            print(f"[cascade] layer1: fetch ok — {len(html)} chars")
        except Exception as exc:
            layer1_error = str(exc)
            print(f"[cascade] layer1: fetch failed — {layer1_error}")

        if html:
            block = detect_gateway_block(html)
            if block:
                print(f"[cascade] layer1: gateway block → {block!r}")
                return CascadeResult(
                    success=False, path="blocked",
                    content=f"blocked: {block}",
                    final_url=final_url,
                )
            content = _trafilatura_extract(html)
            useful = _is_useful_extract(content, goal)
            print(f"[cascade] layer1: extract → {len(content)} chars, useful={useful}")
            if useful:
                print("[cascade] layer1: ✓ sufficient — returning extract")
                return CascadeResult(
                    success=True, path="extract",
                    content=content, final_url=final_url,
                )
            print("[cascade] layer1: insufficient — escalating to a11y")
        else:
            print("[cascade] layer1: no HTML — escalating to a11y")

        # ── Layer 2b: A11yDriver ────────────────────────────────────────────
        print("[cascade] layer2b: running A11yDriver ...")
        a11y_result = await self._drive(A11yDriver, url, goal, self.max_steps_a11y)
        print(f"[cascade] layer2b: done — success={a11y_result.success}  "
              f"note={a11y_result.note!r}")

        if getattr(a11y_result, "gateway_blocked", False):
            print("[cascade] layer2b: gateway block after JS render")
            return CascadeResult(
                success=False, path="blocked",
                content=f"blocked: {a11y_result.note}",
                final_url=getattr(a11y_result, "final_url", url),
            )

        if a11y_result.success:
            print("[cascade] layer2b: ✓ a11y succeeded")
            return CascadeResult(
                success=True, path="a11y",
                content=getattr(a11y_result, "extracted", "") or "",
                final_url=getattr(a11y_result, "final_url", url),
                actions=getattr(a11y_result, "actions", []),
            )
        print("[cascade] layer2b: a11y failed — escalating to vision")

        # ── Layer 3: SetOfMarksDriver ───────────────────────────────────────
        print("[cascade] layer3: running SetOfMarksDriver ...")
        vis_result = await self._drive(SetOfMarksDriver, url, goal, self.max_steps_vision)
        print(f"[cascade] layer3: done — success={vis_result.success}  "
              f"note={vis_result.note!r}")

        if getattr(vis_result, "gateway_blocked", False):
            print("[cascade] layer3: gateway block after JS render")
            return CascadeResult(
                success=False, path="blocked",
                content=f"blocked: {vis_result.note}",
                final_url=getattr(vis_result, "final_url", url),
            )

        if vis_result.success:
            print("[cascade] layer3: ✓ vision succeeded")
            return CascadeResult(
                success=True, path="vision",
                content=getattr(vis_result, "extracted", "") or "",
                final_url=getattr(vis_result, "final_url", url),
                actions=getattr(vis_result, "actions", []),
            )

        last_err = vis_result.note or a11y_result.note or layer1_error or "all layers exhausted"
        print(f"[cascade] ✗ all layers exhausted — {last_err}")
        return CascadeResult(
            success=False, path="blocked",
            content=f"All layers exhausted: {last_err}",
            final_url=url,
            error=last_err,
        )

    async def _drive(self, DriverCls, url: str, goal: str, max_steps: int):
        artifacts_dir = None
        if self.artifacts_dir:
            sub = Path(self.artifacts_dir) / DriverCls.LAYER_NAME
            sub.mkdir(parents=True, exist_ok=True)
            artifacts_dir = str(sub)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            ctx = await browser.new_context(
                viewport={"width": 1366, "height": 900},
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                locale="en-US",
            )
            await ctx.add_init_script(
                "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
            )
            page = await ctx.new_page()
            try:
                print(f"[cascade._drive] {DriverCls.__name__}: navigating to {url} ...")
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                print(f"[cascade._drive] {DriverCls.__name__}: loaded — {page.url}")

                rendered_html = await page.content()
                block = detect_gateway_block(rendered_html)
                if block:
                    print(f"[cascade._drive] {DriverCls.__name__}: gateway block → {block!r}")
                    await browser.close()
                    out = DriverResult(
                        success=False,
                        note=f"gateway_blocked ({block}) at {page.url}",
                    )
                    out.gateway_blocked = True
                    return out

                await asyncio.sleep(1.0)
                cfg = DriverConfig(
                    goal=goal, max_steps=max_steps, max_failures=3,
                    artifacts_dir=artifacts_dir,
                )
                drv = DriverCls(page, self._client, cfg)
                result = await drv.run()

                result.final_url = page.url
                result.extracted = ""
                result.actions = [
                    {"turn": s.turn, "actions": s.actions, "outcome": s.outcome}
                    for s in drv.steps
                ]
                result.turns = len(drv.steps)

                # Extract final page content
                try:
                    page_html = await page.content()
                    traf = _trafilatura_extract(page_html)
                    if len(traf) >= 500:
                        result.extracted = traf[:15000]
                        print(f"[cascade._drive] extraction=trafilatura {len(result.extracted)} chars")
                    else:
                        inner = await page.inner_text("body")
                        result.extracted = inner[:15000]
                        print(f"[cascade._drive] extraction=inner_text {len(result.extracted)} chars")
                except Exception as exc:
                    print(f"[cascade._drive] extraction failed — {exc}")

                return result
            finally:
                await browser.close()
