import asyncio
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from browser_agent.graph import build_graph
from browser_agent.state import BrowserState

load_dotenv()
console = Console()

_NODE_STYLE: dict[str, str] = {
    "planner":       "cyan",
    "browser_skill": "blue",
    "answer":        "magenta",
    "extractor":     "green",
    "comparator":    "green",
    "responder":     "white",
}


def _print_update(node_name: str, update: dict) -> None:
    colour = _NODE_STYLE.get(node_name, "white")

    if node_name == "planner":
        task_type = update.get("task_type", "")
        url = update.get("browser_url", "")
        console.print(
            f"[{colour}]plan[/{colour}] [{task_type}] → {url[:80]}"
        )

    elif node_name == "browser_skill":
        path = update.get("browser_path", "?")
        content_len = len(update.get("browser_content", ""))
        actions = update.get("browser_actions", [])
        console.print(
            f"[{colour}]browse[/{colour}] path={path}  "
            f"turns={len(actions)}  content={content_len} chars"
        )

    elif node_name == "answer":
        console.print(f"[{colour}]answer[/{colour}] ◉ generating answer from page content ...")

    elif node_name == "extractor":
        items = update.get("extracted_items", [])
        console.print(f"[{colour}]extract[/{colour}] ⬇ {len(items)} product(s) found")

    elif node_name == "comparator":
        console.print(f"[{colour}]compare[/{colour}] ⚖ generating comparison ...")


async def run_agent(task: str) -> None:
    console.print(Panel(f"[bold]{task}[/bold]", title="[cyan]Browser Agent[/cyan]", expand=False))
    console.print()

    initial_state: BrowserState = {
        "task": task,
        "browser_url": "",
        "browser_goal": "",
        "task_type": "",
        "browser_content": "",
        "browser_path": "",
        "browser_actions": [],
        "extracted_items": [],
        "comparison_result": "",
        "final_answer": "",
        "status": "planning",
    }

    graph = build_graph()
    final_answer: str = ""

    async for chunk in graph.astream(initial_state, stream_mode="updates"):
        for node_name, update in chunk.items():
            _print_update(node_name, update)
            if "final_answer" in update and update["final_answer"]:
                final_answer = update["final_answer"]

    console.print()
    console.print(Rule(style="dim"))
    console.print()

    if final_answer:
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
