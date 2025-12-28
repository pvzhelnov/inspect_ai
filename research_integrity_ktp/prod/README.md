# Researcher Profiling Agent - Production System

Production-ready, scalable researcher profiling system using Inspect AI and SQLite.

## Features

- **Scalable**: Handle millions of researcher samples
- **Parallel Processing**: Multi-worker batch processing
- **Database Logging**: All LLM requests and responses logged to SQLite
- **Resume Capability**: Interrupt and resume processing at any time
- **Rate Limiting**: Configurable API rate limits
- **Progress Tracking**: Real-time progress monitoring
- **Error Recovery**: Automatic retries with exponential backoff
- **Web Caching**: Cache web search and browser results

## Architecture

```
prod/
├── db/                     # Database layer
│   ├── schema.sql         # SQLite schema
│   ├── database.py        # Database manager with connection pooling
│   └── repositories.py    # Repository classes for CRUD operations
├── models/                 # Data models
│   └── schemas.py         # Pydantic models for all entities
├── agents/                 # AI agents
│   └── researcher_profiling.py  # Main research agent
├── utils/                  # Utilities
│   └── batch_processor.py # Batch processing engine
└── scripts/                # Executable scripts
    ├── run_batch.py       # Main batch processing script
    ├── generate_samples.py # Generate sample data
    └── query_db.py        # Query database for results
```

## Database Schema

### Core Tables

- **researchers**: Sample data for researchers to profile
- **research_iterations**: Logs of each 8-step research iteration
- **llm_requests**: All LLM API calls with tokens and costs
- **orchestrator_decisions**: Orchestrator agent decisions
- **processing_batches**: Batch job tracking
- **web_search_cache**: Cached web search results
- **browser_cache**: Cached browsed pages
- **system_metrics**: System performance metrics

## Quick Start

### 1. Install Dependencies

```bash
cd research_integrity_ktp
pip install -e ".[dev]"
```

### 2. Set Environment Variables

Create a `.env` file:

```bash
OPENROUTER_API_KEY=your_api_key_here
TAVILY_API_KEY=your_tavily_key_here  # Optional, for web search
```

### 3. Prepare Input Data

Create a JSONL file with researcher samples:

```jsonl
{"researcher_id": "hinton_001", "name": "Geoffrey Hinton", "field": "Artificial Intelligence", "known_info": {"affiliation": "University of Toronto"}}
{"researcher_id": "lecun_002", "name": "Yann LeCun", "field": "Computer Science", "known_info": {"affiliation": "NYU"}}
{"researcher_id": "bengio_003", "name": "Yoshua Bengio", "field": "Machine Learning", "known_info": {"affiliation": "Mila"}}
```

Or generate sample data:

```bash
python scripts/generate_samples.py --output samples.jsonl --count 1000
```

### 4. Run Batch Processing

```bash
python scripts/run_batch.py \
    --input samples.jsonl \
    --batch-name batch_001 \
    --workers 4 \
    --model openrouter/qwen/qwen3-coder:free \
    --max-iterations 5
```

### 5. Monitor Progress

```bash
# View progress
tail -f batch_processing.log

# Query database
python scripts/query_db.py --stats

# Resume interrupted batch
python scripts/run_batch.py --resume --batch-name batch_001
```

## Usage Examples

### Basic Usage

```bash
# Process 100 researchers with 4 workers
python scripts/run_batch.py \
    --input researchers.jsonl \
    --batch-name my_batch \
    --workers 4
```

### With Web Tools

```bash
# Enable web_search and web_browser tools
python scripts/run_batch.py \
    --input researchers.jsonl \
    --batch-name web_enabled_batch \
    --workers 2 \
    --enable-tools
```

### Load Only (No Processing)

```bash
# Just load samples into database
python scripts/run_batch.py \
    --input large_dataset.jsonl \
    --batch-name load_only \
    --load-only
```

### Resume After Interruption

```bash
# Resume from where you left off
python scripts/run_batch.py \
    --batch-name my_batch \
    --resume
```

### Custom Configuration

```bash
# Custom model, batch size, and iterations
python scripts/run_batch.py \
    --input researchers.jsonl \
    --batch-name custom_batch \
    --workers 8 \
    --batch-size 200 \
    --model openrouter/anthropic/claude-3.5-sonnet \
    --max-iterations 10
```

## Querying Results

### Python API

```python
from pathlib import Path
from research_integrity_ktp.prod.db.database import Database, DatabaseConfig
from research_integrity_ktp.prod.db.repositories import (
    ResearcherRepository,
    ResearchIterationRepository,
)

# Connect to database
db_config = DatabaseConfig(db_path=Path("data/researcher_profiles.db"))
db = Database(db_config)

# Get repositories
researcher_repo = ResearcherRepository(db)
iteration_repo = ResearchIterationRepository(db)

# Query researchers
researchers = researcher_repo.get_pending(limit=10)
for researcher in researchers:
    print(f"{researcher.name} - {researcher.status}")

# Get iterations for a researcher
iterations = iteration_repo.get_by_researcher("hinton_001")
for iteration in iterations:
    if iteration.step_8_extracted_data:
        print(f"Iteration {iteration.iteration_number}:")
        print(f"  Publications: {len(iteration.step_8_extracted_data.publications)}")
        print(f"  H-index: {iteration.step_8_extracted_data.h_index}")
```

### SQL Queries

```sql
-- Count researchers by status
SELECT status, COUNT(*) as count
FROM researchers
GROUP BY status;

-- Get LLM request statistics
SELECT
    request_type,
    COUNT(*) as requests,
    SUM(tokens_total) as total_tokens,
    SUM(cost_usd) as total_cost,
    AVG(latency_ms) as avg_latency
FROM llm_requests
WHERE status = 'completed'
GROUP BY request_type;

-- Get researchers with most iterations
SELECT
    researcher_id,
    COUNT(*) as iterations
FROM research_iterations
GROUP BY researcher_id
ORDER BY iterations DESC
LIMIT 10;
```

## Performance Tuning

### Database Optimization

The system uses WAL mode for better concurrency:

```python
db_config = DatabaseConfig(
    db_path=Path("data/profiles.db"),
    pool_size=20,           # Increase for more workers
    cache_size=-128000,     # 128MB cache
    journal_mode="WAL",
    synchronous="NORMAL",
)
```

### Batch Processing

```python
processor = BatchProcessor(
    db_path=db_path,
    batch_name="large_batch",
    worker_count=8,                # More workers = faster
    batch_size=500,                # Larger batches = fewer DB ops
    max_retries=3,
    rate_limit_per_minute=120,     # Adjust based on API limits
)
```

## Scaling to Millions

### Hardware Recommendations

For 1M+ researchers:
- **CPU**: 8+ cores (for parallel workers)
- **RAM**: 16GB+ (for database cache)
- **Disk**: SSD recommended (WAL mode benefits from fast I/O)

### Processing Strategy

```bash
# Split large datasets into multiple batches
split -l 100000 million_researchers.jsonl batch_

# Process each batch separately
for batch in batch_*; do
    python scripts/run_batch.py \
        --input "$batch" \
        --batch-name "$(basename $batch)" \
        --workers 8
done
```

### Database Maintenance

```sql
-- Vacuum database periodically
VACUUM;

-- Analyze for query optimization
ANALYZE;

-- Check database size
SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size();
```

## Monitoring and Logging

### Log Files

- `batch_processing.log`: Main processing log
- SQLite database: All structured data

### Metrics

```python
# Get batch statistics
stats = processor.get_stats()
print(f"Progress: {stats['progress_percentage']:.2f}%")
print(f"Rate: {stats['rate_per_second']:.2f} samples/sec")

# Get LLM statistics
llm_stats = agent.llm_repo.get_stats()
print(f"Total tokens: {llm_stats['total_tokens']}")
print(f"Total cost: ${llm_stats['total_cost']:.2f}")
```

## Troubleshooting

### Database Locked Error

If you see "database is locked" errors:
- Reduce `worker_count`
- Increase `timeout` in DatabaseConfig
- Check that WAL mode is enabled

### Out of Memory

- Reduce `batch_size`
- Reduce `cache_size` in DatabaseConfig
- Process in smaller batches

### API Rate Limits

- Adjust `rate_limit_per_minute`
- Use multiple API keys with load balancing
- Reduce `worker_count`

## License

MIT License
