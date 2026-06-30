import base64
from typing import Any
from playwright.async_api import Page

INTERACTIVE_ROLES = {
    "button", "link", "checkbox", "textbox", "combobox",
    "menuitem", "tab", "radio", "searchbox", "option",
}


async def get_page_snapshot(page: Page) -> str:
    """Return a text representation of the page's accessibility tree."""
    try:
        tree = await page.accessibility.snapshot()
        if tree:
            return _flatten_tree(tree).strip()
        return ""
    except Exception:
        return ""


def _flatten_tree(node: dict[str, Any], depth: int = 0) -> str:
    lines: list[str] = []
    indent = "  " * depth
    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")
    checked = node.get("checked")

    if role and (name or value):
        line = f"{indent}[{role}] {name}"
        if value:
            line += f" = {value!r}"
        if checked is not None:
            line += f" ({'✓' if checked else '○'})"
        lines.append(line)

    for child in node.get("children", []):
        child_text = _flatten_tree(child, depth + 1)
        if child_text.strip():
            lines.append(child_text)

    return "\n".join(lines)


def _count_interactive(node: dict[str, Any]) -> int:
    count = 1 if node.get("role", "") in INTERACTIVE_ROLES else 0
    for child in node.get("children", []):
        count += _count_interactive(child)
    return count


async def is_snapshot_sparse(page: Page, threshold: int = 12) -> bool:
    """Returns True when the a11y tree has fewer than threshold interactive elements."""
    try:
        tree = await page.accessibility.snapshot()
        if not tree:
            return True
        return _count_interactive(tree) < threshold
    except Exception:
        return True


async def take_screenshot_b64(page: Page) -> str:
    """Capture a full-page screenshot and return it as a base64 PNG string."""
    data = await page.screenshot(type="png", full_page=False)
    return base64.b64encode(data).decode()
