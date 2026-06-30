from typing import TypedDict


class BrowserState(TypedDict):
    task: str

    # Set by planner
    browser_url: str        # URL for the cascade to visit
    browser_goal: str       # Specific goal for the driver (what to navigate/extract)
    task_type: str          # "shopping" | "informational"

    # Set by browser_skill after cascade runs
    browser_content: str    # Extracted page text from cascade
    browser_path: str       # "extract" | "a11y" | "vision" | "blocked"
    browser_actions: list   # Step records from driver turns

    # Shopping pipeline
    extracted_items: list[dict]
    comparison_result: str

    # Final output
    final_answer: str
    status: str             # "planning" | "browsing" | "done"
