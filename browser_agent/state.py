from typing import TypedDict, Annotated
import operator


class BrowserState(TypedDict):
    task: str
    # action_history accumulates across all nodes via operator.add
    action_history: Annotated[list[dict], operator.add]
    current_url: str
    page_snapshot: str          # accessibility tree text
    use_vision: bool            # True when a11y tree is sparse → trigger llava
    pending_action: dict | None # action decided by planner, consumed by executor
    extracted_items: list[dict] # structured product data collected so far
    comparison_result: str      # markdown comparison from comparator
    final_answer: str           # formatted output for user
    error_count: int
    iteration: int
    status: str                 # "browsing" | "extracting" | "done"
