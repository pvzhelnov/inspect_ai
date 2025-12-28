-- Production database schema for researcher profiling agent
-- Designed to handle millions of samples and LLM requests

-- Researchers table: stores sample data for researchers to profile
CREATE TABLE IF NOT EXISTS researchers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    researcher_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    field TEXT,
    known_info TEXT, -- JSON blob of known information
    status TEXT DEFAULT 'pending', -- pending, processing, completed, failed
    priority INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_researchers_status ON researchers(status);
CREATE INDEX IF NOT EXISTS idx_researchers_priority ON researchers(priority DESC);
CREATE INDEX IF NOT EXISTS idx_researchers_created ON researchers(created_at);

-- Research iterations: logs each research iteration
CREATE TABLE IF NOT EXISTS research_iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    researcher_id TEXT NOT NULL,
    iteration_number INTEGER NOT NULL,
    step_1_reflection TEXT, -- JSON
    step_2_previous_review TEXT, -- JSON
    step_3_location_selection TEXT, -- JSON
    step_4_language_selection TEXT, -- JSON
    step_5_web_search TEXT, -- JSON
    step_6_search_strategy TEXT, -- JSON
    step_7_browsing TEXT, -- JSON
    step_8_extracted_data TEXT, -- JSON
    status TEXT DEFAULT 'pending', -- pending, completed, failed
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (researcher_id) REFERENCES researchers(researcher_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_iterations_researcher ON research_iterations(researcher_id);
CREATE INDEX IF NOT EXISTS idx_iterations_status ON research_iterations(status);
CREATE INDEX IF NOT EXISTS idx_iterations_number ON research_iterations(researcher_id, iteration_number);

-- LLM requests: logs all API calls for debugging and cost tracking
CREATE TABLE IF NOT EXISTS llm_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT UNIQUE NOT NULL,
    researcher_id TEXT,
    iteration_id INTEGER,
    request_type TEXT NOT NULL, -- research_iteration, orchestrator_decision
    model_name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT,
    tokens_prompt INTEGER,
    tokens_completion INTEGER,
    tokens_total INTEGER,
    cost_usd REAL DEFAULT 0.0,
    latency_ms INTEGER,
    status TEXT DEFAULT 'pending', -- pending, success, failed
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (iteration_id) REFERENCES research_iterations(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_llm_requests_status ON llm_requests(status);
CREATE INDEX IF NOT EXISTS idx_llm_requests_researcher ON llm_requests(researcher_id);
CREATE INDEX IF NOT EXISTS idx_llm_requests_created ON llm_requests(created_at);
CREATE INDEX IF NOT EXISTS idx_llm_requests_type ON llm_requests(request_type);

-- Orchestrator decisions: tracks orchestrator agent decisions
CREATE TABLE IF NOT EXISTS orchestrator_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    researcher_id TEXT NOT NULL,
    iteration_number INTEGER NOT NULL,
    continue_research BOOLEAN NOT NULL,
    rationale TEXT NOT NULL,
    completeness_score REAL NOT NULL,
    filled_fields TEXT, -- JSON array
    missing_fields TEXT, -- JSON array
    next_focus_areas TEXT, -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (researcher_id) REFERENCES researchers(researcher_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_orchestrator_researcher ON orchestrator_decisions(researcher_id);
CREATE INDEX IF NOT EXISTS idx_orchestrator_created ON orchestrator_decisions(created_at);

-- Processing batches: tracks batch jobs for parallelization
CREATE TABLE IF NOT EXISTS processing_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_name TEXT UNIQUE NOT NULL,
    total_samples INTEGER NOT NULL,
    processed_samples INTEGER DEFAULT 0,
    successful_samples INTEGER DEFAULT 0,
    failed_samples INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending', -- pending, running, completed, failed, cancelled
    worker_count INTEGER DEFAULT 1,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_batches_status ON processing_batches(status);
CREATE INDEX IF NOT EXISTS idx_batches_created ON processing_batches(created_at);

-- Web search cache: cache web search results to avoid duplicates
CREATE TABLE IF NOT EXISTS web_search_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_query TEXT NOT NULL,
    language TEXT NOT NULL,
    results TEXT NOT NULL, -- JSON blob of search results
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    UNIQUE(search_query, language)
);

CREATE INDEX IF NOT EXISTS idx_search_cache_query ON web_search_cache(search_query, language);
CREATE INDEX IF NOT EXISTS idx_search_cache_expires ON web_search_cache(expires_at);

-- Browser cache: cache browsed pages to avoid re-fetching
CREATE TABLE IF NOT EXISTS browser_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_browser_cache_url ON browser_cache(url);
CREATE INDEX IF NOT EXISTS idx_browser_cache_expires ON browser_cache(expires_at);

-- System metrics: track system performance and statistics
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_unit TEXT,
    metadata TEXT, -- JSON blob for additional info
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_metrics_name ON system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_created ON system_metrics(created_at);
