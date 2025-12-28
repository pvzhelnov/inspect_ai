# Researcher Profiling Agent - Complete Implementation

Single comprehensive implementation with screenshot-based decision making.

## Architecture

```
1. LLM Planning          → Decide what to search
2. NATIVE Tavily Search  → RAW results saved
3. LLM URL Selection     → Choose best URL
4. NATIVE Playwright     → Screenshot + HTML dump
5. LLM Strategy Decision → text_search OR web_browser_tool
6a. Text Search Path     → grep/regex in HTML
6b. Browser Tool Path    → Inspect AI web_browser agentically
7. LLM Extraction        → Extract from results
8. LLM Orchestrator      → Continue research?
```

## Key Features

✅ **Screenshot-Based Decisions** - LLM sees actual page screenshot
✅ **Dual Execution Paths** - Text search OR web browser tool
✅ **Native Playwright** - Real browser screenshots and HTML dumps
✅ **Native Tavily API** - Direct API calls, RAW results saved
✅ **Persistent Storage** - All tool outputs saved to disk
✅ **Mock Mode** - Tests both paths without LLM API costs

## Usage

### Real Mode (with LLM calls)
```bash
TAVILY_API_KEY=xxx python researcher_profiling_agent.py
```

### Mock Mode (LLM mocked, tools run for real)
```bash
MOCK_MODE=1 python researcher_profiling_agent.py
```

## Testing Strategy

### Mock Mode Tests Both Paths

**Mock fixtures control which path:**
- Set `strategy: "text_search"` → Tests text search path
- Set `strategy: "web_browser_tool"` → Tests browser tool path

**Currently configured:** `web_browser_tool` (tests that path)

### What Gets Tested in Mock Mode

1. ✅ **Tavily Search** - REAL API call, RAW results saved
2. ✅ **Playwright** - REAL browser, screenshot + HTML saved
3. ✅ **Strategy Decision** - Mock LLM returns fixture
4. ✅ **Web Browser Tool** - REAL Inspect AI tool execution
5. ✅ **Extraction** - Mock LLM returns fixture
6. ✅ **Orchestrator** - Mock LLM returns fixture

### Critical Bug That Was Fixed

**Problem:**
```python
async def use_web_browser_tool_agentically(...):  # WRONG!
    log = eval(browser_task(), model=model)[0]
```

**Error:**
```
RuntimeWarning: coroutine 'App.run.<locals>.run_app' was never awaited
```

**Cause:** Function marked `async def` but `eval()` is synchronous, creating unawaited coroutine

**Fix:**
```python
def use_web_browser_tool_agentically(...):  # CORRECT!
    log = eval(browser_task(), model=model)[0]
```

**How It Was Caught:**
- Changed MOCK_MODE to use `web_browser_tool` instead of `text_search`
- Removed mock skip: browser tool now ACTUALLY executes in MOCK_MODE
- Running mock mode revealed the async/await bug immediately

## Files Generated

### Mock Mode Output
```
logs/researcher_profiling/
├── step2_raw_tavily_results.json    # RAW Tavily search response
├── step3_screenshot.png             # Actual browser screenshot
├── step3_html_dump.html             # Full HTML from browser
├── step5_browser_tool_results.json  # Web browser tool results
└── pipeline_summary.json            # Complete run summary
```

### Real Mode Output
Same files, but with actual LLM responses instead of fixtures.

## Code Structure

```python
# Step models with @model_validator for NATIVE execution
class Step2_TavilySearch(BaseModel):
    search_query: str

    @model_validator(mode="after")
    def execute_search(self):
        client = TavilyClient(...)
        response = client.search(...)  # NATIVE API call
        # Save RAW results
        return self

# Strategy execution
if strategy == "text_search":
    # Path A: Grep/regex in HTML
    text_search_in_html(html, patterns)
else:
    # Path B: Web browser tool (synchronous!)
    use_web_browser_tool_agentically(url, goal, model)
```

## Dependencies

```toml
playwright>=1.40.0     # Native browser automation
tavily-python>=0.7.0   # Native Tavily API client
inspect-ai>=0.3.0      # LLM evaluation framework
```

## Common Issues

### "No sandbox environment" Error

**Error:**
```
ProcessLookupError: No sandbox environment has been provided
```

**Cause:** Web browser tool requires sandbox but none configured

**Solution:** Error is handled gracefully, results still saved. In production, configure sandbox:
```python
@task
def browser_task():
    return Task(
        ...,
        sandbox="docker",  # or "local"
    )
```

### Network Errors in Playwright

**Error:**
```
Page.goto: net::ERR_NAME_NOT_RESOLVED
```

**Solution:** Playwright configured with:
```python
browser = await p.chromium.launch(
    args=['--no-sandbox', '--disable-setuid-sandbox']
)
context = await browser.new_context(
    ignore_https_errors=True,
    bypass_csp=True
)
```

## Development Notes

### Why No Separate Test File?

Mock mode IS the test. It exercises both execution paths:
- Change `Step4_StrategyDecision` fixture to switch paths
- All tools execute for real (Tavily, Playwright, web_browser)
- Only LLM is mocked with fixtures
- Catches bugs that unit tests miss (like async/await issues)

### Pydantic @model_validator Pattern

```python
class ToolExecutor(BaseModel):
    query: str
    results_: list = Field(default_factory=list, exclude=True)

    @model_validator(mode="after")
    def execute_tool(self):
        # Execute AFTER Pydantic validates JSON from LLM
        self.results_ = native_function_call(self.query)
        return self
```

This pattern:
- LLM generates JSON
- Pydantic validates structure
- @model_validator executes native Python function
- Results available for next LLM call
- NO tool calling features from Inspect AI

## Comparison: Text Search vs Web Browser Tool

### Text Search Path
- ✅ Fast (no LLM calls for navigation)
- ✅ Works on static HTML
- ✅ No sandbox needed
- ❌ Can't handle JavaScript
- ❌ Can't click/navigate

### Web Browser Tool Path
- ✅ Handles JavaScript
- ✅ Can navigate/click
- ✅ Agentic (LLM decides actions)
- ❌ Slower (multiple LLM calls)
- ❌ Requires sandbox
- ❌ More expensive

## Production Deployment

For production at scale, see:
- `prod/agents/researcher_profiling_native.py` - Production agent
- `prod/db/` - Database layer
- `prod/scripts/run_batch.py` - Batch processing
- `ARCHITECTURE.md` - Complete architectural guide
