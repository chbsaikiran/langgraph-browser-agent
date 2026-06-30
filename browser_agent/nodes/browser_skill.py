"""LangGraph node that runs the 4-layer browser cascade."""
from browser_agent.state import BrowserState
from browser_agent.browser.cascade import BrowserCascade


async def browser_skill_node(state: BrowserState) -> dict:
    url = state.get("browser_url", "")
    goal = state.get("browser_goal", "")

    if not url:
        return {
            "browser_content": "",
            "browser_path": "blocked",
            "browser_actions": [],
            "status": "done",
        }

    cascade = BrowserCascade()
    result = await cascade.run(url, goal)

    return {
        "browser_content": result.content,
        "browser_path": result.path,
        "browser_actions": result.actions,
        "status": "extracting" if result.success else "done",
    }
