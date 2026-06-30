from playwright.async_api import TimeoutError as PWTimeout
from browser_agent.state import BrowserState
from browser_agent.browser.controller import BrowserController
from browser_agent.browser.snapshot import get_page_snapshot, is_snapshot_sparse

_NAV_TIMEOUT = 15_000   # ms for page navigation
_ACT_TIMEOUT = 4_000    # ms for element interactions
_SETTLE_TIMEOUT = 5_000 # ms to wait for networkidle after each action


async def executor_node(state: BrowserState) -> dict:
    action = state["pending_action"]
    page = await BrowserController.get_page()
    result = "success"

    act = action.get("action", "")
    target = action.get("target") or ""
    value = action.get("value") or ""

    try:
        if act == "navigate":
            await page.goto(target, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT)

        elif act == "click":
            ok = await _smart_click(page, target)
            if not ok:
                result = f"failed: element not found '{target}'"

        elif act == "type":
            ok = await _smart_type(page, target, value)
            if not ok:
                result = f"failed: field not found '{target}'"
            else:
                # Submit the input — works for search boxes; harmless on plain fields
                await page.keyboard.press("Enter")
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=8_000)
                except PWTimeout:
                    pass

        elif act == "key":
            await page.keyboard.press(target or value)

        elif act == "scroll":
            delta = -700 if (value or "down").lower() == "up" else 700
            await page.mouse.wheel(0, delta)
            await page.wait_for_timeout(600)

        elif act == "set_price":
            # target = max price (e.g. "80000"), value = min price (optional, defaults to "1")
            ok = await _set_price_range(page, min_price=value or "1", max_price=target)
            if not ok:
                result = f"failed: could not set price filter (max={target})"

        # Let the page settle after any action
        try:
            await page.wait_for_load_state("networkidle", timeout=_SETTLE_TIMEOUT)
        except PWTimeout:
            pass

    except Exception as exc:
        result = f"error: {str(exc)[:120]}"

    # Read updated page state
    snapshot = await get_page_snapshot(page)
    sparse = await is_snapshot_sparse(page, threshold=12)

    history_entry = {
        "action": act,
        "target": target or None,
        "value": value or None,
        "reason": action.get("reason"),
        "result": result,
        "url": page.url,
    }

    return {
        "current_url": page.url,
        "page_snapshot": snapshot,
        "use_vision": sparse and result == "success",
        "action_history": [history_entry],
    }


async def _smart_click(page, text: str) -> bool:
    # Step 1: try Playwright locators — scroll element into view BEFORE clicking
    # (critical for Amazon's sidebar filters which live in a scrollable container)
    locator_factories = [
        lambda: page.get_by_text(text, exact=True).first,
        lambda: page.get_by_role("link", name=text).first,
        lambda: page.get_by_role("checkbox", name=text).first,
        lambda: page.get_by_role("button", name=text).first,
        lambda: page.get_by_role("tab", name=text).first,
        lambda: page.get_by_label(text).first,
        lambda: page.get_by_text(text, exact=False).first,
    ]
    for get_loc in locator_factories:
        try:
            loc = get_loc()
            await loc.scroll_into_view_if_needed(timeout=2_000)
            await loc.click(timeout=_ACT_TIMEOUT)
            return True
        except Exception:
            continue

    # Step 2: JavaScript fallback — walks the real DOM so it finds elements
    # that the a11y tree misses (Amazon filter spans, price range anchors, etc.)
    safe = text.replace("\\", "\\\\").replace("'", "\\'")
    try:
        clicked = await page.evaluate(f"""() => {{
            const tags = 'a, button, label, span, li, input[type="checkbox"]';
            const els = [...document.querySelectorAll(tags)];
            const el =
                els.find(e => e.textContent.trim() === '{safe}') ||
                els.find(e => e.textContent.trim().includes('{safe}'));
            if (el) {{
                el.scrollIntoView({{behavior: 'instant', block: 'center'}});
                el.click();
                return true;
            }}
            return false;
        }}""")
        if clicked:
            return True
    except Exception:
        pass

    # Step 3: force-click the first exact text match (bypasses visibility check)
    try:
        await page.get_by_text(text, exact=False).first.click(
            timeout=_ACT_TIMEOUT, force=True
        )
        return True
    except Exception:
        pass

    return False


async def _set_price_range(page, min_price: str, max_price: str) -> bool:
    """Fill min/max price inputs on any e-commerce site and click Go."""
    max_strategies = [
        lambda: page.get_by_placeholder("Max").first.fill(max_price, timeout=_ACT_TIMEOUT),
        lambda: page.get_by_placeholder("max").first.fill(max_price, timeout=_ACT_TIMEOUT),
        lambda: page.get_by_label("Max").first.fill(max_price, timeout=_ACT_TIMEOUT),
        lambda: page.locator("input[placeholder*='max' i]").first.fill(max_price, timeout=_ACT_TIMEOUT),
        lambda: page.locator("input[aria-label*='max' i]").first.fill(max_price, timeout=_ACT_TIMEOUT),
        lambda: page.locator("input[id*='high' i]").first.fill(max_price, timeout=_ACT_TIMEOUT),
        lambda: page.locator("input[name*='high' i]").first.fill(max_price, timeout=_ACT_TIMEOUT),
        # Amazon specifically uses id="high-price"
        lambda: page.locator("#high-price").first.fill(max_price, timeout=_ACT_TIMEOUT),
    ]

    filled = False
    for strategy in max_strategies:
        try:
            await strategy()
            filled = True
            break
        except Exception:
            continue

    if not filled:
        return False

    # Try to also fill min price
    min_strategies = [
        lambda: page.get_by_placeholder("Min").first.fill(min_price, timeout=_ACT_TIMEOUT),
        lambda: page.locator("input[placeholder*='min' i]").first.fill(min_price, timeout=_ACT_TIMEOUT),
        lambda: page.locator("input[id*='low' i]").first.fill(min_price, timeout=_ACT_TIMEOUT),
        lambda: page.locator("#low-price").first.fill(min_price, timeout=_ACT_TIMEOUT),
    ]
    for strategy in min_strategies:
        try:
            await strategy()
            break
        except Exception:
            continue

    # Click "Go" or press Enter to apply
    go_strategies = [
        lambda: page.get_by_role("button", name="Go").first.click(timeout=2_000),
        lambda: page.get_by_text("Go", exact=True).first.click(timeout=2_000),
        lambda: page.locator("input[type='submit']").first.click(timeout=2_000),
        lambda: page.keyboard.press("Enter"),
    ]
    for strategy in go_strategies:
        try:
            await strategy()
            return True
        except Exception:
            continue

    return filled


async def _smart_type(page, field: str, text: str) -> bool:
    strategies = [
        lambda: page.get_by_placeholder(field).first.fill(text, timeout=_ACT_TIMEOUT),
        lambda: page.get_by_label(field).first.fill(text, timeout=_ACT_TIMEOUT),
        lambda: page.get_by_role("searchbox", name=field).first.fill(text, timeout=_ACT_TIMEOUT),
        lambda: page.get_by_role("textbox", name=field).first.fill(text, timeout=_ACT_TIMEOUT),
        # fallback: find any searchbox/textbox on the page
        lambda: page.get_by_role("searchbox").first.fill(text, timeout=_ACT_TIMEOUT),
        lambda: page.get_by_role("textbox").first.fill(text, timeout=_ACT_TIMEOUT),
    ]
    for strategy in strategies:
        try:
            await strategy()
            return True
        except Exception:
            continue
    return False
