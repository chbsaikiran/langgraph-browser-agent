import asyncio
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from browser_agent.browser.controller import BrowserController
from browser_agent.graph import build_graph
from browser_agent.state import BrowserState

load_dotenv()
console = Console()

# Node label → display colour
_NODE_STYLE: dict[str, str] = {
    "planner":    "cyan",
    "executor":   "blue",
    "vision":     "yellow",
    "answer":     "magenta",
    "extractor":  "green",
    "comparator": "green",
    "responder":  "white",
}


def _print_update(node_name: str, update: dict) -> None:
    colour = _NODE_STYLE.get(node_name, "white")

    if node_name == "planner":
        action = update.get("pending_action") or {}
        act = action.get("action", "?").upper()
        target = action.get("target") or ""
        reason = action.get("reason") or ""
        iteration = update.get("iteration", "")
        console.print(
            f"[dim][{iteration:>2}][/dim] [{colour}]{act:<10}[/{colour}]"
            f" [bold]{target[:60]}[/bold]"
            + (f" — [dim]{reason[:60]}[/dim]" if reason else "")
        )

    elif node_name == "executor":
        history = update.get("action_history") or []
        if history:
            last = history[-1]
            ok = "✓" if last.get("result") == "success" else "✗"
            console.print(
                f"         [{colour}]{ok} {last.get('result', '')[:80]}[/{colour}]"
            )

    elif node_name == "vision":
        console.print(f"         [{colour}]◎ llava:7b vision analysis running...[/{colour}]")

    elif node_name == "answer":
        console.print(f"         [{colour}]◉ reading page and answering question...[/{colour}]")

    elif node_name == "extractor":
        items = update.get("extracted_items") or []
        console.print(f"         [{colour}]⬇ extracted {len(items)} product(s)[/{colour}]")

    elif node_name == "comparator":
        console.print(f"         [{colour}]⚖ generating comparison...[/{colour}]")


async def run_agent(task: str) -> None:
    console.print(Panel(f"[bold]{task}[/bold]", title="[cyan]Browser Agent[/cyan]", expand=False))
    console.print()

    initial_state: BrowserState = {
        "task": task,
        "action_history": [],
        "current_url": "",
        "page_snapshot": "",
        "use_vision": False,
        "pending_action": None,
        "extracted_items": [],
        "comparison_result": "",
        "final_answer": "",
        "error_count": 0,
        "iteration": 0,
        "status": "browsing",
    }

    graph = build_graph()
    final_answer: str = ""

    try:
        async for chunk in graph.astream(initial_state, stream_mode="updates"):
            for node_name, update in chunk.items():
                _print_update(node_name, update)
                if "final_answer" in update and update["final_answer"]:
                    final_answer = update["final_answer"]

    finally:
        await BrowserController.close()

    console.print()
    console.print(Rule(style="dim"))
    console.print()

    if final_answer:
        # Guard: any node could theoretically return a list of content blocks
        if isinstance(final_answer, list):
            final_answer = "\n".join(
                p.get("text", str(p)) if isinstance(p, dict) else str(p)
                for p in final_answer if p
            )
        console.print(Markdown(str(final_answer)))
    else:
        console.print("[red]Agent finished but produced no output.[/red]")


def main() -> None:
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        console.print("[bold cyan]Browser Agent[/bold cyan] — LangGraph + Playwright")
        console.print("[dim]LLM: Gemini   Vision fallback: llava:7b (Ollama)[/dim]\n")
        task = console.input("[bold]Task:[/bold] ").strip()
        if not task:
            console.print("[red]No task provided.[/red]")
            sys.exit(1)

    asyncio.run(run_agent(task))


if __name__ == "__main__":
    main()
