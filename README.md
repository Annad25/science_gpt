# Science GPT — Multi-Agent RAG Pipeline over Scientific Data

A production-grade, multi-agent retrieval-augmented generation (RAG) system that ingests heterogeneous scientific sources, routes queries to specialised agents, synthesises answers via LangChain Deep Agents, and verifies outputs against source material.

## Architecture

```
User Query
    │
    ▼
┌──────────┐     ┌─────────────────────┐     ┌───────────┐     ┌───────────┐
│  Router   │────▶│  Retrieval Agents   │────▶│  Reasoner │────▶│ Evaluator │
│  Agent    │     │  (Vector+Structured)│     │ (Deep Agent)│    │ (LLM Judge)│
└──────────┘     └─────────────────────┘     └───────────┘     └───────────┘
    │                     │                        │                  │
    │              ┌──────┴──────┐           ┌─────┴─────┐           │
    │              ▼             ▼           ▼           ▼           │
    │         Qdrant        WHO CSV     Calculator  Code Exec       │
    │        (vectors)     (Pandas)    Summarizer                   │
    └───────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    JSON Response + Sources + Latency
```

### Pipeline Stages

1. **Router Agent** — Classifies queries via LLM (with keyword-heuristic fallback) into `text`, `structured`, or `hybrid` and dispatches to the appropriate retriever(s).

2. **Retrieval Agents** — Run in parallel via `asyncio.gather()`:
   - **Vector Retriever** — Encodes the query with `all-MiniLM-L6-v2` and searches Qdrant for top-k semantically similar chunks from arXiv papers.
   - **Structured Retriever** — Uses LLM-based query parsing to extract filters/aggregations, then queries a WHO health statistics DataFrame via Pandas.

3. **Reasoning Agent** — A LangChain Deep Agent (`deepagents.create_deep_agent`) synthesises the final answer using retrieved context + tools (calculator, code executor, summariser). Falls back to direct OpenAI if Deep Agents is unavailable.

4. **Evaluator Agent** — LLM-as-judge that verifies every claim in the answer against source material, producing a verdict (`supported`, `partially_supported`, `not_supported`) with issues list.

## Features Implemented

### Level 1 — Core
- [x] Ingestion pipeline for two source types (arXiv API + WHO CSV)
- [x] Retrieval agent with routing to vector and structured search
- [x] Reasoning agent with 3 tools beyond retrieval (calculator, summariser, sandboxed Python)
- [x] Query API (`POST /query`) returning answer, sources, and latency breakdown

### Level 2 — Scalability
- [x] **Concurrency isolation** — Each request gets a unique `trace_id` via `contextvars`; all agents are stateless and async; no shared mutable state between requests
- [x] **Intelligent caching** — In-memory TTL cache with hash-based keys. `POST /cache/invalidate` for manual busting. `POST /ingest` re-indexes and auto-invalidates the cache, preventing stale results after data refresh
- [x] **Entity traversal** — The Structured Retriever can parse complex queries (group-by aggregations, multi-column filters), traversing relationships between countries, years, and health metrics

### Level 3 — Robustness
- [x] **Hallucination detection** — Evaluator Agent (LLM-as-judge) checks every claim against sources. Answers that fail verification are replaced with a safe fallback
- [x] **Systematic evaluation** — `python -m app.evaluation.systematic` runs 8 diverse test questions through the pipeline and produces a JSON report with per-question verdicts, routing accuracy, and latency statistics
- [x] **Sandboxed code execution** — AST-based static analysis blocks dangerous imports/builtins. Code runs in a subprocess with empty environment, timeout, and output cap

## Setup & Configuration

### Prerequisites
- Python 3.11+
- An OpenAI API key

### Quick Start

```bash
# Clone the repo
git clone <repo-url> && cd science_gpt

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY

# Run the service
python run.py
```

The service starts on `http://localhost:8000`. On first startup, it:
1. Downloads the `all-MiniLM-L6-v2` embedding model (~80MB, cached after first run)
2. Fetches ~15 papers from arXiv (takes 30-60s)
3. Chunks, embeds, and indexes them into Qdrant (in-memory)
4. Loads the WHO CSV dataset

### Docker

```bash
docker compose up --build
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `LLM_MODEL` | `gpt-4o` | Model for reasoning/evaluation |
| `QDRANT_MODE` | `memory` | `memory`, `docker`, or `cloud` |
| `RETRIEVAL_TOP_K` | `5` | Number of vector search results |
| `CHUNK_SIZE` | `800` | Characters per text chunk |
| `CACHE_TTL_SECONDS` | `600` | Cache entry lifetime |
| `ARXIV_MAX_PAPERS` | `15` | Papers to fetch per query |

See `.env.example` for the full list.

## API Reference

### `POST /query`
```json
{
  "query": "What is the life expectancy in Japan and how does AI help healthcare?"
}
```

Response:
```json
{
  "answer": "Based on the available sources...",
  "query_type": "hybrid",
  "sources": [
    {"source": "arXiv:...", "chunk_id": "...", "relevance_score": 0.87, "text_snippet": "..."}
  ],
  "evaluation": {
    "verdict": "supported",
    "confidence": 0.95,
    "issues": []
  },
  "latency": {
    "routing_ms": 450.2,
    "retrieval_ms": 120.5,
    "reasoning_ms": 2100.3,
    "evaluation_ms": 800.1,
    "total_ms": 3471.1
  },
  "cached": false
}
```

### `POST /ingest` — Re-run ingestion, invalidate cache
### `POST /cache/invalidate?query=...` — Bust specific or all cache entries
### `GET /health` — Liveness check
### `GET /stats` — Pipeline statistics

## Running Tests

```bash
pytest tests/ -v
```

## Running Systematic Evaluation

```bash
python -m app.evaluation.systematic
```

Outputs a JSON report with per-question verdicts, routing accuracy, and latency p95.

## Key Architectural Decisions

### Why explicit agent orchestration over monolithic chains?
We deliberately avoid LangChain's `RetrievalQA` or `AgentExecutor` black-box chains. Instead, each stage (Route → Retrieve → Reason → Evaluate) is an explicit Python async call. This makes the pipeline transparent, testable, and debuggable — you can see exactly what each agent does in the logs. We use LangChain Deep Agents only for the reasoning step, where its planning and context-management capabilities add genuine value.

### Why Qdrant in-memory over ChromaDB?
Qdrant's in-memory mode gives us a production-grade vector store API (with payload filtering, hybrid search support, and well-typed Python client) without requiring Docker or a running service. ChromaDB is simpler but lacks Qdrant's filtering and production features. Switching to Docker or Cloud mode is a one-line config change (`QDRANT_MODE=docker`).

### Why arXiv abstracts instead of full PDFs?
Full PDF parsing is expensive, error-prone, and slow. arXiv abstracts are dense, high-quality summaries that capture the key findings of each paper. For a 24-hour build, this gives 90% of the value at 10% of the complexity. The ingestion pipeline is designed to be extensible — adding full-text PDF parsing is a matter of adding a PDF loader to `arxiv_loader.py`.

### Why AST-based code sandboxing instead of Docker containers?
Docker-in-Docker or container-per-execution adds significant infrastructure complexity. Our AST-based approach statically validates code before execution, blocks dangerous imports/builtins at parse time, and runs in a subprocess with an empty environment and strict timeout. This is sufficient for the computation-only use case (math, statistics, data analysis) while being deployable anywhere Python runs.

### Why LLM-as-judge for evaluation?
Research on LLM hallucination detection shows that using a second LLM as a critic (with explicit instructions to check source attribution) catches a significant fraction of fabricated claims. This is more flexible than rule-based checking and doesn't require maintaining a separate NLI model. The trade-off is cost (an extra LLM call per query), which is acceptable for correctness-critical scientific QA.

## LLM Conversation

This project was designed and built through an extensive conversation with Claude (Anthropic), covering:
- Architecture design: choosing explicit agent orchestration over monolithic chains
- Deep Agents SDK integration: researching and integrating `deepagents.create_deep_agent`
- Data source selection: evaluating arXiv API vs. local PDFs, WHO CSV format
- Sandboxing strategy: designing AST-based code validation vs. Docker isolation
- Hallucination prevention: implementing the LLM-as-judge pattern

## What I Would Do Differently With More Time

1. **Learned router** — Replace the LLM+heuristic router with a fine-tuned classification model (or use the Deep Agent framework itself to spawn sub-agents dynamically).
2. **Full-text PDF ingestion** — Add a PDF parser (e.g., `pymupdf4llm`) to index full paper bodies, not just abstracts, with section-aware chunking.
3. **Hybrid reranking** — Add a cross-encoder reranker (e.g., `ms-marco-MiniLM-L-6-v2`) after initial retrieval to improve precision.
4. **Graph-based entity linking** — Use Neo4j to model relationships between papers, authors, concepts, and health metrics, enabling multi-hop reasoning.
5. **Streaming responses** — Use FastAPI's `StreamingResponse` with LangGraph's streaming to show intermediate agent steps in real-time.
6. **Redis caching** — Replace in-memory cache with Redis for persistence across restarts and distributed deployments.

## Known Limitations

- **In-memory Qdrant** — Vector data is lost on restart. Use `QDRANT_MODE=docker` for persistence.
- **arXiv rate limits** — The arXiv API is rate-limited; large ingestion runs may be throttled.
- **No authentication** — The API has no auth layer. Add API key middleware for production.
- **Embedding model size** — `all-MiniLM-L6-v2` is fast but not state-of-the-art. Consider `nomic-embed-text` or OpenAI embeddings for higher quality.
