# LangGraph Browser Agent

A fully autonomous web browsing agent built with **LangGraph** and **Playwright**. Give it a task in plain English — it navigates, interacts with, and extracts information from any website, then delivers a structured answer.

**Primary LLM:** Google Gemini (via API key)  
**Vision fallback:** `llava:7b` running locally via Ollama  
**Browser engine:** Playwright (Chromium, headless)

---

## What It Can Do

```
"Go to amazon.in, find laptops under ₹80,000, and compare the top 3"
    → Navigates Amazon, applies price filter, extracts product specs,
      generates a comparison table with a recommendation

"Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his
 birth date, death date, and three key contributions to information theory"
    → Fetches the page, extracts the requested facts, returns a formatted answer
```

The agent handles two task classes:

| Task Class | Trigger | Flow |
|---|---|---|
| **Informational** | Reading facts, articles, answering questions from a page | Planner → Cascade → Answer |
| **Shopping / Comparison** | Searching products, filtering, comparing | Planner → Cascade → Extract → Compare → Respond |

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER TASK (plain English)                   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LANGGRAPH STATE MACHINE                        │
│                                                                     │
│  ┌──────────┐    ┌───────────────┐    ┌──────────┐                 │
│  │ PLANNER  │───▶│ BROWSER SKILL │───▶│  ANSWER  │──▶ END          │
│  │ (Gemini) │    │  (Cascade)    │    │ (Gemini) │                 │
│  └──────────┘    └───────┬───────┘    └──────────┘                 │
│                          │                                          │
│                          │ shopping                                 │
│                          ▼                                          │
│                   ┌──────────────┐   ┌──────────────┐              │
│                   │  EXTRACTOR   │──▶│  COMPARATOR  │──▶ RESPOND   │
│                   │  (Gemini)    │   │  (Gemini)    │              │
│                   └──────────────┘   └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## LangGraph State Machine — Detailed Flow

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                         ▼
              ┌────────────────────┐
              │      PLANNER       │
              │  Gemini analyzes   │
              │  task → outputs:   │
              │  • browser_url     │
              │  • browser_goal    │
              │  • task_type       │
              └────────┬───────────┘
                       │
                       ▼
           ┌─────────────────────────┐
           │     BROWSER SKILL       │
           │   4-Layer Cascade runs  │
           │   (see cascade diagram) │
           │   outputs:              │
           │   • browser_content     │
           │   • browser_path        │
           │   • browser_actions     │
           └──────────┬──────────────┘
                      │
            ┌─────────┴──────────┐
            │  route_after_browser│
            └─────────┬──────────┘
                      │
       ┌──────────────┼───────────────┐
       │              │               │
  "informational"  "shopping"     "blocked"
       │              │               │
       ▼              ▼               ▼
  ┌────────┐   ┌──────────┐   ┌──────────────┐
  │ ANSWER │   │ EXTRACTOR│   │  RESPONDER   │
  │        │   │          │   │ (error msg)  │
  └───┬────┘   └────┬─────┘   └──────┬───────┘
      │             │                │
      │             ▼                │
      │       ┌──────────┐           │
      │       │COMPARATOR│           │
      │       └────┬─────┘           │
      │            │                 │
      │            ▼                 │
      │       ┌──────────┐           │
      │       │ RESPONDER│           │
      │       └────┬─────┘           │
      │            │                 │
      └────────────┴─────────────────┘
                   │
                   ▼
                 ┌─────┐
                 │ END │
                 └─────┘
```

### State Object (`browser_agent/state.py`)

The entire agent state flows through a single `TypedDict` that every LangGraph node reads from and writes to:

```python
class BrowserState(TypedDict):
    task: str              # Original user task (never changes)

    # Set by PLANNER
    browser_url: str       # URL to visit
    browser_goal: str      # What the browser driver must accomplish
    task_type: str         # "shopping" | "informational"

    # Set by BROWSER SKILL after cascade
    browser_content: str   # Full extracted text from the page
    browser_path: str      # "extract"|"a11y"|"vision"|"blocked"
    browser_actions: list  # Per-turn step records from the driver

    # Set by EXTRACTOR (shopping flow only)
    extracted_items: list[dict]  # Structured product dicts

    # Set by COMPARATOR (shopping flow only)
    comparison_result: str

    # Final output (set by ANSWER or RESPONDER)
    final_answer: str
    status: str
```

---

## Core Engine: The 4-Layer Browser Cascade

This is the most important subsystem. It is called once per agent run from `browser_skill_node` and attempts four progressively more expensive strategies to retrieve and process a web page. It stops at the **first layer that succeeds**.

```
┌──────────────────────────────────────────────────────────────┐
│                    4-LAYER CASCADE                           │
│                  browser/cascade.py                          │
│                                                              │
│  INPUT: (url: str, goal: str)                                │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ LAYER 1 — HTTP Extract (no browser, no LLM)          │   │
│  │  httpx.get(url) → trafilatura.extract(html)          │   │
│  │  ✓ If content ≥ 200 chars AND goal is read-only      │   │
│  │    → return immediately  (Wikipedia, articles, etc.) │   │
│  │  ✗ If goal requires interaction (filter/search/click)│   │
│  │    → escalate to Layer 2b                            │   │
│  └────────────────────────┬─────────────────────────────┘   │
│                           │ (insufficient or interactive)    │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ LAYER 2b — A11y Text Driver (Gemini, no screenshot)  │   │
│  │  Playwright launches browser → navigates to URL      │   │
│  │  Per-turn loop (up to 15 turns):                     │   │
│  │    enumerate_interactives(page) → text legend        │   │
│  │    Gemini decides actions from legend only           │   │
│  │    dispatch action → wait → next turn                │   │
│  │  ✓ Driver signals done(success=true)                 │   │
│  │    → extract page text → return                      │   │
│  │  ✗ Driver fails / step cap reached                   │   │
│  │    → escalate to Layer 3                             │   │
│  └────────────────────────┬─────────────────────────────┘   │
│                           │ (a11y driver failed)             │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ LAYER 3 — Set-of-Marks Vision Driver (llava:7b)      │   │
│  │  Playwright launches browser → navigates to URL      │   │
│  │  Per-turn loop (up to 12 turns):                     │   │
│  │    enumerate_interactives(page) → legend             │   │
│  │    take_screenshot() + annotate() → marked PNG       │   │
│  │    llava:7b decides actions from image + legend      │   │
│  │    dispatch action → wait → next turn                │   │
│  │  ✓ Driver signals done(success=true)                 │   │
│  │    → extract page text → return                      │   │
│  │  ✗ All layers exhausted                              │   │
│  │    → return path="blocked"                           │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  OUTPUT: CascadeResult(success, path, content, final_url)    │
└──────────────────────────────────────────────────────────────┘
```

### Why 4 Layers?

Each layer is a cost/capability trade-off:

| Layer | Cost | Speed | Capability |
|-------|------|-------|------------|
| 1 — HTTP extract | Free | ~1s | Read-only, JS-unrendered pages |
| 2b — A11y text | Gemini text call/turn | ~10–30s | Full interaction, no vision model |
| 3 — Set-of-Marks | llava vision call/turn | ~30–90s | JS-heavy sites, visual disambiguation |

---

## The Per-Turn Driver Loop

Both A11y and Vision drivers share the same per-turn control loop implemented in `BaseDriver`:

```
┌──────────────────────────────────────────────────────────────┐
│                   PER-TURN DRIVER LOOP                       │
│               browser/driver.py — BaseDriver                 │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Turn N                                             │   │
│   │                                                     │   │
│   │  1. enumerate_interactives(page)                    │   │
│   │     ↓ runs _ENUMERATE_JS on the live page           │   │
│   │     ↓ returns PageSnapshot with Element list        │   │
│   │                                                     │   │
│   │  2. _decide(snap, turn)   ←── LAYER SPECIFIC        │   │
│   │     A11y:   build text legend → Gemini /chat        │   │
│   │     Vision: screenshot → annotate → llava /vision   │   │
│   │     ↓ LLM returns JSON: {thinking, actions:[...]}   │   │
│   │                                                     │   │
│   │  3. for each action in actions:                     │   │
│   │     _dispatch(action, page, snap)                   │   │
│   │     ├── click(mark)  → page.mouse.click(el.cx,cy)  │   │
│   │     ├── type(mark,v) → click + keyboard.type(v)    │   │
│   │     ├── key(v)       → keyboard.press(v)           │   │
│   │     ├── scroll(dir)  → page.mouse.wheel(dx,dy)     │   │
│   │     ├── drag(x,y)    → mouse.move+down+up          │   │
│   │     ├── wait(s)      → asyncio.sleep(s)            │   │
│   │     └── done(ok,msg) → exit loop                   │   │
│   │                                                     │   │
│   │  4. Record StepRecord → append to steps list        │   │
│   │                                                     │   │
│   │  ─────────────────────────────────────────────────  │   │
│   │  if done(success=true)  → DriverResult(success)     │   │
│   │  if failures ≥ 3        → DriverResult(give up)     │   │
│   │  if turn ≥ max_steps    → DriverResult(step cap)    │   │
│   │  else                   → Turn N+1                  │   │
│   └─────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### Action Vocabulary (shared by both drivers)

The LLM is given this exact schema and must return JSON matching it:

```json
{
  "thinking": "1-2 sentences of reasoning",
  "actions": [
    { "type": "click",  "mark": 7 },
    { "type": "type",   "mark": 3,  "value": "laptops", "clear": true },
    { "type": "key",    "value": "Enter" },
    { "type": "scroll", "direction": "down", "amount": 600 },
    { "type": "drag",   "from_x": 100, "from_y": 200, "to_x": 400, "to_y": 200 },
    { "type": "wait",   "seconds": 1.5 },
    { "type": "done",   "success": true, "note": "products visible" }
  ]
}
```

The `mark` field is the **element ID** from the interactive element legend — a number like `[7]` that maps to a specific button, link, or input on the page.

---

## Interactive Element Enumeration (`browser/dom.py`)

Before every driver turn, the entire interactive element set is re-enumerated from the live DOM via a single JavaScript evaluation:

```
┌──────────────────────────────────────────────────────────────┐
│              _ENUMERATE_JS  (runs in browser context)        │
│                                                              │
│  Selects elements by:                                        │
│  ├── HTML tags: a[href], button, input, textarea, select     │
│  ├── ARIA roles: button, link, tab, menuitem, textbox, ...   │
│  ├── cursor:pointer (catches JS-driven clickables)           │
│  └── contenteditable, onclick, label[for], summary           │
│                                                              │
│  Filters:                                                    │
│  ├── Drops SVG primitives (path, rect, circle, g, …)        │
│  ├── Dedupes by outermost-ancestor (click the wrapper,       │
│  │   not the inner icon)                                     │
│  └── Drops off-screen and visibility:hidden elements         │
│                                                              │
│  Each element returns:                                       │
│  { id, tag, role, name, x, y, w, h }  (all in CSS pixels)  │
│                                                              │
│  Name resolution order:                                      │
│  aria-label → aria-labelledby → innerText → value →         │
│  placeholder → title → alt → data-tooltip → data-testid     │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
              PageSnapshot.legend() produces:
              [1]<a>Sign in</a>
              [2]<input role="searchbox">Search Amazon.in</input>
              [3]<button>Go</button>
              [4]<a>Laptops</a>
              ...
```

This text legend is what the A11y driver's Gemini call receives. The Vision driver also gets this legend alongside the annotated screenshot.

---

## Screenshot Annotation (`browser/highlight.py`)

The Set-of-Marks Vision driver overlays numbered, colored, dashed boxes on every interactive element before sending the image to llava:7b:

```
┌──────────────────────────────────────────────────────────────┐
│                  annotate(png, elements, dpr)                │
│                                                              │
│  Raw screenshot (device pixels) ─────────────────────┐      │
│                                                       │      │
│  For each Element:                                    │      │
│    CSS_x × dpr → device_x  (handles 2x Retina)       │      │
│    CSS_y × dpr → device_y                             │      │
│                                                       │      │
│  Color by tag:                                        │      │
│    a      → Blue   (#2E86C1)                          │      │
│    button → Green  (#27AE60)                          │      │
│    input  → Orange (#E67E22)                          │      │
│    select → Purple (#9B59B6)                          │      │
│    other  → Red    (#C0392B)                          │      │
│                                                       │      │
│  Draw dashed rectangle border                         │      │
│  Draw filled badge: ┌──┐                              │      │
│                     │ 7│  ← element id                │      │
│                     └──┘                              │      │
│                                                       │      │
│  Output PNG → base64 data URL → llava:7b              │      │
└──────────────────────────────────────────────────────────────┘

  Before annotation:          After annotation:
  ┌─────────────────┐         ┌─────────────────┐
  │  Search Amazon  │         │ ┌1┐Search Amazon │
  │  [Go]  [Cart]   │         │ ─── [Go]²[Cart]³ │
  │  Laptops        │         │  Laptops⁴        │
  └─────────────────┘         └─────────────────┘
```

---

## LLM Client (`browser/llm_client.py`)

A thin adapter that presents the same `.chat()` / `.vision()` interface to both drivers, routing calls to the right model:

```
┌──────────────────────────────────────────────────────────────┐
│                       LLMClient                              │
│                                                              │
│  .chat(prompt, system, schema)                               │
│     │                                                        │
│     └──▶  ChatGoogleGenerativeAI (Gemini)                    │
│           response_mime_type="application/json"              │
│           → force JSON output mode                           │
│           → _extract_json(text) → parsed dict                │
│                                                              │
│  .vision(image_data_url, prompt, system, schema)             │
│     │                                                        │
│     └──▶  ChatOllama (llava:7b, local Ollama)                │
│           HumanMessage with image_url + text content         │
│           → _extract_json(text) → parsed dict                │
│                                                              │
│  Returns: LLMResult(parsed, text, provider, model,           │
│                     latency_ms, input_tokens, output_tokens)  │
└──────────────────────────────────────────────────────────────┘

  JSON extraction (_extract_json):
  1. Strip markdown code fences (```json ... ```)
  2. Try json.loads() on the cleaned text
  3. Regex search for {...} and try json.loads()
  4. Return None if all fail → driver records "parse error"
```

---

## Planner Node (`nodes/planner.py`)

The planner runs **once** at the start of every agent invocation. It converts the raw user task into a structured plan that the browser cascade can execute:

```
┌──────────────────────────────────────────────────────────────┐
│                     PLANNER NODE                             │
│                                                              │
│  Input:  state["task"]  (plain English)                      │
│                                                              │
│  Gemini.with_structured_output(BrowserPlan)                  │
│                                                              │
│  Output: BrowserPlan                                         │
│    url       → The exact URL to visit                        │
│    goal      → Specific instructions for the driver          │
│    task_type → "shopping" | "informational"                  │
│    reason    → Brief explanation                             │
│                                                              │
│  Examples:                                                   │
│                                                              │
│  Task: "find laptops under ₹80,000 on amazon"               │
│  ──────────────────────────────────────────────              │
│  url: "https://www.amazon.in/s?k=laptops&rh=p_36%3A100-8000000"  │
│  goal: "Amazon search results with price filter applied.     │
│         Products are visible. Signal done."                  │
│  task_type: "shopping"                                       │
│                                                              │
│  Task: "fetch https://en.wikipedia.org/wiki/Claude_Shannon   │
│         and tell me his birth date"                          │
│  ──────────────────────────────────────────────              │
│  url: "https://en.wikipedia.org/wiki/Claude_Shannon"         │
│  goal: "extract birth date, death date, key contributions"   │
│  task_type: "informational"                                  │
└──────────────────────────────────────────────────────────────┘
```

**Amazon Price Filter Encoding (built into planner prompt):**

```
₹80,000  →  max_paise = 80,000 × 100 = 8,000,000
URL parameter: rh=p_36%3A100-8000000
Full URL: https://www.amazon.in/s?k=laptops&rh=p_36%3A100-8000000

Amazon's p_36 parameter takes PAISE (not rupees).
1 rupee = 100 paise. The URL already encodes the price filter,
so the A11y driver only needs to wait for results and signal done.
```

---

## Downstream Nodes

### Answer Node (`nodes/answer.py`) — Informational tasks

```
browser_content (up to 20,000 chars of page text)
        │
        ▼
┌───────────────────────────────┐
│  ANSWER NODE                  │
│                               │
│  System: ANSWER_SYSTEM prompt │
│  User:   task + page_text     │
│                               │
│  Gemini → final_answer        │
│                               │
│  Rules enforced by prompt:    │
│  • Answer from page only      │
│  • No prior knowledge         │
│  • Exact dates/names/quotes   │
│  • Markdown formatting        │
└───────────────────────────────┘
        │
        ▼
    state["final_answer"] → END
```

### Extractor Node (`nodes/extractor.py`) — Shopping tasks

```
browser_content (up to 12,000 chars)
        │
        ▼
┌───────────────────────────────┐
│  EXTRACTOR NODE               │
│                               │
│  Gemini.with_structured_output│
│  (ProductList pydantic model) │
│                               │
│  Extracts per product:        │
│  • name, price, brand         │
│  • processor, ram, storage    │
│  • display, rating, reviews   │
│                               │
│  Returns: list[Product] ≤ 5   │
└───────────────────────────────┘
        │
        ▼
    state["extracted_items"]
```

### Comparator Node (`nodes/comparator.py`)

```
extracted_items (top 3 products as JSON)
        │
        ▼
┌───────────────────────────────┐
│  COMPARATOR NODE              │
│                               │
│  Gemini generates:            │
│  1. Markdown comparison table │
│     Feature | P1 | P2 | P3    │
│  2. Key Differences (bullets) │
│  3. Recommendation (2 lines)  │
└───────────────────────────────┘
        │
        ▼
    state["comparison_result"]
```

### Responder Node (`nodes/responder.py`)

```
comparison_result (or extracted_items if no comparison)
        │
        ▼
┌───────────────────────────────┐
│  RESPONDER NODE               │
│                               │
│  Assembles final markdown:    │
│  • "# Result: <task>"         │
│  • comparison table           │
│  • browser path + turn count  │
│                               │
│  Also handles blocked path:   │
│  • Returns error explanation  │
└───────────────────────────────┘
        │
        ▼
    state["final_answer"] → END
```

---

## Complete File Structure

```
Browser_Agent/
├── main.py                          # CLI entry point, streams graph updates
├── pyproject.toml                   # Dependencies (uv managed)
│
└── browser_agent/
    ├── state.py                     # BrowserState TypedDict
    ├── graph.py                     # LangGraph StateGraph + routing functions
    │
    ├── browser/                     # Browser automation layer
    │   ├── dom.py                   # JS element enumeration → PageSnapshot
    │   ├── highlight.py             # Pillow screenshot annotation (set-of-marks)
    │   ├── llm_client.py            # LLMClient: Gemini (text) + llava (vision)
    │   ├── driver.py                # BaseDriver, A11yDriver, SetOfMarksDriver
    │   └── cascade.py               # 4-layer cascade orchestrator
    │
    ├── nodes/                       # LangGraph nodes
    │   ├── planner.py               # Task → BrowserPlan (url + goal + type)
    │   ├── browser_skill.py         # Calls cascade, writes results to state
    │   ├── answer.py                # Informational: page text → answer
    │   ├── extractor.py             # Shopping: page text → structured products
    │   ├── comparator.py            # Products JSON → comparison table
    │   └── responder.py             # Final answer assembly
    │
    └── prompts/
        └── templates.py             # All system prompts
```

---

## Data Flow: Shopping Task End-to-End

```
User: "Go to amazon.in, find laptops under ₹80,000, compare top 3"
  │
  ▼
PLANNER (Gemini)
  url  = "https://www.amazon.in/s?k=laptops&rh=p_36%3A100-8000000"
  goal = "Amazon search with price filter active. Products visible. Signal done."
  type = "shopping"
  │
  ▼
BROWSER SKILL → BrowserCascade.run(url, goal)
  │
  ├── Layer 1: httpx GET amazon.in → HTML returned but
  │   goal contains "filter" → _is_useful_extract=False → escalate
  │
  └── Layer 2b: A11yDriver (Playwright + Gemini)
        Turn 1: elements enumerated (search box, nav links, product cards)
                Gemini: "Products are already visible with price filter in URL.
                         Signal done."
                → done(success=True, note="product listings visible")
        Extract: trafilatura(page_html) → 8,000 chars of product text
        CascadeResult(path="a11y", content="ASUS Vivobook...\n₹52,990...")
  │
  ▼
EXTRACTOR (Gemini structured output)
  browser_content (product text) → ProductList
  extracted_items = [
    { name: "ASUS Vivobook 15", price: 52990, ram: "16 GB", ... },
    { name: "HP Pavilion x360",  price: 64999, processor: "i5-13...", ... },
    { name: "Lenovo IdeaPad",    price: 45990, storage: "512 GB SSD", ... },
  ]
  │
  ▼
COMPARATOR (Gemini)
  | Feature   | ASUS Vivobook  | HP Pavilion | Lenovo IdeaPad |
  |-----------|----------------|-------------|----------------|
  | Price     | ₹52,990        | ₹64,999     | ₹45,990        |
  | ...                                                        |
  Recommendation: "Lenovo IdeaPad offers the best value..."
  │
  ▼
RESPONDER → final_answer (markdown) → console output
```

---

## Data Flow: Informational Task End-to-End

```
User: "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and
       tell me his birth date, death date, and 3 key contributions"
  │
  ▼
PLANNER (Gemini)
  url  = "https://en.wikipedia.org/wiki/Claude_Shannon"
  goal = "extract birth date, death date, key contributions"
  type = "informational"
  │
  ▼
BROWSER SKILL → BrowserCascade.run(url, goal)
  │
  └── Layer 1: httpx GET wikipedia → HTML received
      trafilatura extracts ~12,000 chars of article text
      goal has no interactive verbs → _is_useful_extract=True
      → return immediately  (no browser opened!)
      CascadeResult(path="extract", content="Claude Elwood Shannon...")
  │
  ▼
ANSWER (Gemini)
  page_text + task → structured answer
  "Born: April 30, 1916 in Petoskey, Michigan
   Died: February 24, 2001 (aged 84)
   Key contributions:
   1. Information theory (1948 paper 'A Mathematical Theory of Communication')
   ..."
  │
  ▼
  final_answer → END
```

---

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Google Gemini API key
- [Ollama](https://ollama.com/) running locally with `llava:7b` (for vision fallback)

### Install Ollama and pull llava

```bash
# Install Ollama (macOS)
brew install ollama

# Pull llava:7b (vision model used as fallback)
ollama pull llava:7b

# Start Ollama server
ollama serve
```

### Clone and install

```bash
git clone <repo-url>
cd Browser_Agent

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate

uv pip install -e .

# Install Playwright browser binaries
playwright install chromium
```

### Configure environment

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## Running the Agent

### From command line

```bash
# Activate virtual environment
source .venv/bin/activate

# Pass task as argument
python main.py "Go to amazon.in, find laptops under 80000 rupees, compare top 3"

# Or enter task interactively
python main.py
```

### Example commands

```bash
# Shopping comparison
python main.py "Find laptops under ₹60,000 on amazon.in and compare the top 3"

# Wikipedia fact extraction
python main.py "Fetch https://en.wikipedia.org/wiki/Alan_Turing and summarise his life"

# Any informational URL
python main.py "Go to https://en.wikipedia.org/wiki/Python_(programming_language) and list the key features"
```

### Live output while running

The agent streams updates as each node completes:

```
╭─────────────────────────────────────────────────────────────────╮
│ Go to amazon.in, find laptops under ₹80,000, compare top 3     │
╰─────────────────────────────────────────────────────────────────╯

[plan] [shopping] → https://www.amazon.in/s?k=laptops&rh=p_36%3A100-8000000
[browse] path=a11y  turns=2  content=9234 chars
[extract] ⬇ 5 product(s) found
[compare] ⚖ generating comparison ...

──────────────────────────────────────────────────────────────────

# Result: Find laptops under ₹80,000 ...

| Feature   | ASUS Vivobook 15 | HP Pavilion x360 | Lenovo IdeaPad |
...
```

---

## Configuration Reference

| Env Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | (required) | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model ID |

| Cascade Parameter | Default | Where to change |
|---|---|---|
| `max_steps_a11y` | 15 turns | `BrowserCascade.__init__` |
| `max_steps_vision` | 12 turns | `BrowserCascade.__init__` |
| Vision model | `llava:7b` | `LLMClient._get_llava()` |
| Viewport size | 1366×900 | `cascade.py _drive()` |

---

## Dependencies

| Package | Purpose |
|---|---|
| `langgraph` | State machine orchestration |
| `langchain-google-genai` | Gemini LLM calls |
| `langchain-ollama` | llava:7b via local Ollama |
| `langchain-core` | Message types, base LLM interface |
| `playwright` | Headless Chromium browser control |
| `trafilatura` | HTML → clean text extraction (Layer 1) |
| `httpx` | Async HTTP client for Layer 1 fetch |
| `Pillow` | Screenshot annotation with dashed boxes |
| `pydantic` | Structured output models |
| `python-dotenv` | .env file loading |
| `rich` | Terminal markdown rendering |

---

## Design Decisions

**Why LangGraph?**  
LangGraph's `StateGraph` makes the routing logic explicit and inspectable. Every node reads from and writes to a single typed state dict, making debugging straightforward — you can inspect state at any node boundary.

**Why a 4-layer cascade instead of always using vision?**  
Cost and speed. Wikipedia pages load fine with httpx in ~1 second at zero LLM cost. Only when interaction is truly needed (clicking, filtering, searching) does the agent escalate to a browser with an LLM in the loop.

**Why trafilatura for extraction?**  
Trafilatura strips navigation, ads, and sidebars far more reliably than `page.inner_text("body")`. It is used both in Layer 1 (from raw HTML) and after each driver run (from the rendered page) to give downstream nodes clean text.

**Why mark-based actions instead of CSS selectors?**  
Element marks (numeric IDs assigned each turn) are stable for one turn. The LLM picks a number it sees in the legend or screenshot; the dispatcher resolves it to CSS-pixel coordinates. This avoids brittle CSS selectors and works on sites that dynamically generate class names.

**Why Gemini for A11y and llava for Vision?**  
Gemini handles structured JSON output reliably with `response_mime_type="application/json"`. llava:7b is local (free, private) and sufficient for identifying numbered boxes in screenshots. Gemini vision could also be used but llava keeps inference fully local for the visual fallback.
