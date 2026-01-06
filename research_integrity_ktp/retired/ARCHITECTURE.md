# Researcher Profiling System - Architecture

## Correct Implementation Pattern

This document describes the **CORRECT** architecture for the researcher profiling system, as implemented in `test_researcher_profiling_native.py` and `prod/agents/researcher_profiling_native.py`.

## Key Principles

### 1. NO Tool Calling Features

**❌ WRONG** - Using Inspect AI's tool calling:
```python
# DO NOT DO THIS
solver=use_tools([web_search(providers="tavily"), *web_browser()])
```

**✅ CORRECT** - Using NATIVE function execution via Pydantic:
```python
class SearchDecision(BaseModel):
    search_query: str

    @model_validator(mode="after")
    def execute_search(self):
        # NATIVE Tavily API call
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(query=self.search_query)
        self.search_results_ = response.get("results", [])
        return self
```

### 2. Multiple Small LLM Calls

Instead of one large call, use 5-6 small focused calls:

1. **Planning** (256 tokens): LLM decides what to search
2. **Search** (256 tokens): LLM generates query → Pydantic executes NATIVE Tavily API
3. **URL Selection** (256 tokens): LLM selects URL from results
4. **Browsing** (256 tokens): LLM confirms URL → Pydantic executes NATIVE HTTP request
5. **Extraction** (1024 tokens): LLM extracts data from cached page
6. **Orchestrator** (512 tokens): LLM decides if more research needed

This avoids context overload with small models and keeps costs low.

### 3. Persistent Storage of Raw Data

**CRITICAL**: All raw data must be saved persistently!

- ✅ RAW Tavily search results → Save complete JSON response to database
- ✅ FULL web pages → Save complete HTML to database
- ✅ All LLM requests/responses → Save to database with tokens/costs

**Example from test**:
```bash
$ ls -lh logs/researcher_profiling_native/
-rw-r--r-- 1 root root 5.8K step1_raw_search_results.json   # RAW Tavily
-rw-r--r-- 1 root root 171K step3_full_page_content.json    # FULL page
```

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Iteration Loop (max 5 iterations)                              │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STEP 1: Planning                                         │  │
│  │ • LLM generates JSON: {target_website, search_query}     │  │
│  │ • NO function execution yet                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STEP 2: Search (NATIVE execution)                        │  │
│  │ • LLM generates JSON: {search_query}                     │  │
│  │ • Pydantic @model_validator triggers                     │  │
│  │ • NATIVE Tavily API call: TavilyClient().search()        │  │
│  │ • RAW results saved to DB: web_search_cache             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STEP 3: URL Selection                                    │  │
│  │ • LLM receives search results as text                    │  │
│  │ • LLM generates JSON: {selected_url, rationale}          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STEP 4: Browsing (NATIVE execution)                      │  │
│  │ • LLM generates JSON: {url, expected_info}               │  │
│  │ • Pydantic @model_validator triggers                     │  │
│  │ • NATIVE HTTP request: requests.get()                    │  │
│  │ • FULL page saved to DB: browser_cache                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STEP 5: Extraction                                       │  │
│  │ • LLM receives cached page content (first 3000 chars)    │  │
│  │ • LLM generates JSON: {publications, h_index, etc.}      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STEP 6: Orchestrator Decision                            │  │
│  │ • LLM reviews all collected data                         │  │
│  │ • LLM generates JSON: {continue_research, score}         │  │
│  │ • If continue_research=True → next iteration             │  │
│  │ • If continue_research=False → done                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### Pydantic Models with @model_validator

```python
from pydantic import BaseModel, Field, model_validator
from tavily import TavilyClient
import requests

class Step2_Search(BaseModel):
    """LLM generates search query. Pydantic executes NATIVE Tavily API."""

    search_query: str = Field(..., description="Search query")
    max_results: int = Field(default=5)

    # Results populated AFTER execution (exclude from schema)
    search_results_: list[dict] = Field(default_factory=list, exclude=True)

    @model_validator(mode="after")
    def execute_search(self):
        """Automatically execute NATIVE Tavily API when model is validated."""
        api_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=self.search_query,
            max_results=self.max_results
        )

        # Save RAW results
        self.search_results_ = response.get("results", [])

        # Save to database here (in production)
        # save_to_db(response)

        return self

class Step4_Browsing(BaseModel):
    """LLM confirms URL. Pydantic executes NATIVE HTTP request."""

    url: str = Field(..., description="URL to browse")

    # Results populated AFTER execution
    page_content_: str = Field(default="", exclude=True)

    @model_validator(mode="after")
    def execute_browsing(self):
        """Automatically execute NATIVE HTTP request when model is validated."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get(self.url, headers=headers, timeout=30)

        # Save FULL page
        self.page_content_ = response.text

        # Save to database here (in production)
        # save_to_db(response.text)

        return self
```

### Using with Inspect AI

```python
from inspect_ai import Task, eval, task
from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.solver import generate
from inspect_ai.dataset import Sample
from inspect_ai.util import json_schema

@task
def search_task():
    return Task(
        dataset=[Sample(input="Search for Geoffrey Hinton", target="")],
        solver=generate(),  # ONLY generate(), NO use_tools()!
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="Step2_Search",
                json_schema=json_schema(Step2_Search),
                strict=True,
            ),
            max_tokens=256,
        ),
    )

# Execute
model = get_model("openrouter/qwen/qwen3-coder:free")
log = eval(search_task(), model=model)[0]

# When this line runs, Pydantic validates JSON and @model_validator
# automatically executes the NATIVE Tavily API call!
result = Step2_Search.model_validate_json(log.samples[0].output.completion)

# Now result.search_results_ contains the search results
print(f"Found {len(result.search_results_)} results")
```

## Database Schema

### Critical Tables

**web_search_cache** - Stores RAW Tavily responses:
```sql
CREATE TABLE IF NOT EXISTS web_search_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    provider TEXT DEFAULT 'tavily',
    results TEXT NOT NULL,  -- JSON blob with FULL Tavily response
    researcher_id TEXT,
    iteration_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**browser_cache** - Stores FULL web pages:
```sql
CREATE TABLE IF NOT EXISTS browser_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    content TEXT NOT NULL,  -- FULL HTML content
    researcher_id TEXT,
    iteration_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**llm_requests** - Tracks all LLM calls:
```sql
CREATE TABLE IF NOT EXISTS llm_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT UNIQUE NOT NULL,
    researcher_id TEXT,
    iteration_id INTEGER,
    request_type TEXT,  -- 'planning', 'search', 'url_selection', etc.
    model_name TEXT,
    prompt TEXT,
    response TEXT,
    tokens_prompt INTEGER,
    tokens_completion INTEGER,
    latency_ms INTEGER,
    cost_usd REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Testing

### Working Test Implementation

See `research_integrity_ktp/test_researcher_profiling_native.py`:

```bash
$ TAVILY_API_KEY=xxx uv run python research_integrity_ktp/test_researcher_profiling_native.py
```

**Expected output**:
```
======================================================================
NATIVE RESEARCHER PROFILING PIPELINE
(LLM JSON → Pydantic validates → NATIVE function executes)
======================================================================

[STEP 1] LLM generates search query (JSON)
----------------------------------------------------------------------
✓ LLM Output (JSON):
  search_query: Geoffrey Hinton Google Scholar profile publications...
✓ Pydantic executed NATIVE Tavily API call
  Results count: 5

[STEP 2] LLM selects URL from search results (JSON)
----------------------------------------------------------------------
✓ LLM Output (JSON):
  selected_url: https://scholar.google.nl/citations?user=...

[STEP 3] LLM confirms browsing (JSON)
----------------------------------------------------------------------
[Pydantic Validator] Executing NATIVE HTTP request...
  ✓ Status: 200
  ✓ Content length: 169776 characters
  ✓ FULL page saved to: logs/.../step3_full_page_content.json

[STEP 4] LLM extracts data from cached page (JSON)
----------------------------------------------------------------------
✓ LLM Output (JSON):
  h_index: 176
  total_citations: 658923
```

## Production Deployment

### Batch Processing Script

```bash
# Load 1 million samples
python prod/scripts/run_batch.py \
    --input samples_1m.jsonl \
    --batch-name batch_2025_01 \
    --workers 10 \
    --model openrouter/qwen/qwen3-coder:free \
    --max-iterations 5

# Query results
python prod/scripts/query_db.py --stats
python prod/scripts/query_db.py --researcher hinton_001
python prod/scripts/query_db.py --export results.json
```

### Key Features

- ✅ Parallel processing with 10+ workers
- ✅ SQLite with WAL mode for concurrency
- ✅ Connection pooling for thread safety
- ✅ Resume capability after interruption
- ✅ Rate limiting for API calls
- ✅ Comprehensive logging to database
- ✅ Progress tracking per batch
- ✅ Cost tracking per LLM request

## Dependencies

```toml
[project.dependencies]
inspect-ai = ">=0.3.0"
pydantic = ">=2.0.0"
tavily-python = ">=0.7.0"  # NATIVE Tavily client
requests = ">=2.31.0"      # NATIVE HTTP client
python-dotenv = ">=1.0.0"
```

## Common Mistakes to Avoid

### ❌ WRONG: Using use_tools()
```python
solver=[
    use_tools([web_search(providers="tavily"), *web_browser()]),
    generate()
]
```
This uses Inspect AI's tool calling features which are NOT allowed.

### ❌ WRONG: Manual ToolCall creation
```python
tool_call = ToolCall(function="web_search", arguments={"query": "..."})
result = await search_tool(tool_call)
```
This is still using Inspect AI's tool framework.

### ❌ WRONG: Single large LLM call
```python
class ResearchIterationOutput(BaseModel):
    step_1: ...
    step_2: ...
    step_8: ...  # All 8 steps in one call
```
This causes context overload with small models.

### ✅ CORRECT: Pydantic @model_validator
```python
class SearchDecision(BaseModel):
    search_query: str

    @model_validator(mode="after")
    def execute_search(self):
        # NATIVE function call
        client = TavilyClient(...)
        self.results_ = client.search(...)
        return self
```

### ✅ CORRECT: Multiple small calls
```python
step1 = run_planning()  # 256 tokens
step2 = run_search(step1)  # 256 tokens + NATIVE Tavily
step3 = run_url_selection(step2)  # 256 tokens
step4 = run_browsing(step3)  # 256 tokens + NATIVE HTTP
step5 = run_extraction(step4)  # 1024 tokens
```

## Summary

**The Golden Rule**:
> LLM generates JSON → Pydantic validates → @model_validator executes NATIVE Python function → Results saved → Flow to next LLM call

**NO** Inspect AI tool calling features. **YES** NATIVE function execution via Pydantic.
