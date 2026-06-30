from browser_agent.state import BrowserState


async def responder_node(state: BrowserState) -> dict:
    items = state.get("extracted_items", [])
    comparison = state.get("comparison_result", "")
    actions = state.get("browser_actions", [])
    browser_path = state.get("browser_path", "")
    browser_content = state.get("browser_content", "")

    lines: list[str] = [f"# Result: {state['task']}\n"]

    # Blocked / failed cascade
    if browser_path == "blocked":
        lines.append("**Could not complete the task.**\n")
        if browser_content:
            lines.append(browser_content)
        else:
            lines.append(
                "The page was blocked (CAPTCHA, login wall, or all browser layers exhausted). "
                "Try again or use a different source."
            )
        return {"final_answer": "\n".join(lines), "status": "done"}

    if comparison:
        lines.append(comparison)
    elif items:
        lines.append("## Products Found\n")
        for i, item in enumerate(items[:3], 1):
            price = f"₹{item['price']:,.0f}" if item.get("price") else "N/A"
            lines.append(f"**{i}. {item['name']}** — {price}")
            for field in ("processor", "ram", "storage", "display", "rating"):
                if item.get(field):
                    lines.append(f"   - {field.capitalize()}: {item[field]}")
            lines.append("")
    else:
        lines.append(
            "The agent could not extract product data from the page. "
            "The page structure may have changed or the task could not be completed."
        )

    turns = len(actions)
    lines.append(f"\n---\n*Browsed via {browser_path} path in {turns} driver turn(s)*")

    return {"final_answer": "\n".join(lines), "status": "done"}
