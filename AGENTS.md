# Deeper Bot — Agent Context

> This document helps AI agents navigate the codebase effectively. Keep it accurate and update it when architectural decisions change.

## Overview

Deeper Bot is a **Telegram bot for deep research** built around a **ReAct (Reasoning + Acting) agent loop**. The agent uses Tree of Thoughts methodology to carry out multi-step research sessions directly inside Telegram chats.

Key capabilities:

- Multi-turn research agent with tool use (web search, content extraction, user Q&A, status tracking, final report delivery)
- Document uploads: PDF, DOCX, XLSX, PPTX, code files, plain text
- Session persistence via SQLite with automatic context compaction
- SSRF-protected web fetching and search via DuckDuckGo
- LLM-agnostic via LiteLLM (OpenAI, Anthropic, local models, or any OpenAI-compatible endpoint)
- Configurable reasoning effort support (e.g., OpenAI o1/o3 models)

## Tech Stack

| Layer               | Technology                              |
| ------------------- | --------------------------------------- |
| Language            | Python >= 3.14                          |
| Telegram API        | aiogram 3.x (async)                     |
| LLM Gateway         | litellm                                 |
| Web Search          | DuckDuckGo Search (ddgs)                |
| Content Extraction  | trafilatura                             |
| Document Conversion | markitdown                              |
| HTTP Client         | httpx with custom SSRF-safe transport   |
| Configuration       | pydantic-settings                       |
| Database            | aiosqlite (SQLite with WAL mode)        |
| Markdown Processing | mistune (custom Telegram HTML renderer) |
| Build Tool          | uv + hatchling                          |
| Linting             | ruff                                    |
| Testing             | pytest + pytest-asyncio                 |

## Project Structure

```
src/deeper_bot/
├── __main__.py       # Application entry point: sets up Bot, Dispatcher, Router, SessionStore
├── agent.py          # ReAct agent loop: _agent_loop(), run_agent(), typing indicators, error handling
├── bot.py            # Telegram handlers (/clear, /compact, /status), message routing, media group buffering,
│                     # WhitelistMiddleware, content extraction from documents
├── compaction.py     # Context compaction: summarizes old conversation turns to free up context window
├── config.py         # Pydantic Settings with env var / .env loading, validation, API key resolution
├── converter.py      # File-to-markdown conversion: markitdown for office docs, code blocks for code files, plain text
├── llm.py            # LLM client wrapper with retry logic for transient errors
├── prompts.py        # SYSTEM_PROMPT — Tree of Thoughts research methodology and output format
├── session.py        # Session dataclass (state machine), SessionStore (SQLite + in-memory cache with eviction)
└── tools.py          # Agent tool implementations: web_search, web_fetch, ask_user, set_status, finish;
                      # Markdown-to-Telegram-HTML converter; SSRF-safe HTTP client; tool argument validation
```

## Architecture

### ReAct Agent Loop

The core loop lives in [agent.py](src/deeper_bot/agent.py). It operates on a per-chat basis:

1. **Receive user input** → stored in `Session.messages`
2. **Call LLM** with full message history + ephemeral TODO list injection
3. **Process response**:
   - If `tool_calls`: execute each tool, collect results, continue loop
   - If text only: send to user, mark session IDLE
   - If `finish` tool called: deliver report, mark session IDLE
4. **Context window exceeded?** → auto-compact old messages, retry (max 2 compaction attempts)
5. **Iteration cap**: 50 max iterations per research session

### Session State Machine

```
IDLE ──[user sends text]──> RESEARCHING ──[agent asks question]──> AWAITING_ANSWER
  ^                                      │                              │
  │                                      │                              │
  └────[finish / text response]──────────┘                              │
         │                                                              │
         └──────────────────[user replies]──────────────────────────────┘
```

States are defined in `SessionState` ([session.py](src/deeper_bot/session.py)):

- `IDLE`: No active research. New text starts a research session.
- `RESEARCHING`: Agent is running. Incoming messages get a "please wait" reply.
- `AWAITING_ANSWER`: Agent used `ask_user` tool and is blocked on user input.

### Session Persistence

`SessionStore` ([session.py](src/deeper_bot/session.py)) provides:

- SQLite database with WAL mode for durability
- In-memory LRU cache with TTL eviction (1 hour for idle unlocked sessions)
- Max 1000 cached sessions; oldest idle sessions evicted on pressure
- Graceful migration: new columns added via `ALTER TABLE` with duplicate-column handling
- On init: all non-idle states reset to `IDLE` (recovery from crashes)

The `research_start_idx` field separates "old conversation" (eligible for compaction) from the current research turn.

### Context Compaction

When the LLM context window is exceeded, `compact_context()` ([compaction.py](src/deeper_bot/compaction.py)):

1. Collects messages between index 1 and `research_start_idx`
2. Separates previous summaries from raw messages
3. If only summaries exist → deletes them
4. If raw messages exist → sends them to LLM for summarization (utility model, max 1000 tokens)
5. On LLM failure → falls back to keeping last 3 raw messages
6. Replaces old history with `[system, summary] + current_research`
7. Updates `research_start_idx` accordingly

**Double-compaction protection**: If a second compaction occurs with no new raw messages (only a summary), the summary is removed entirely, preventing summary-on-summary bloat.

### Telegram Integration

[bot.py](src/deeper_bot/bot.py) handles all Telegram interactions:

- **Private chat only** — group chats are ignored
- **Media group buffering** — multiple files sent together are collected into a single prompt (1.5s delay)
- **File handling** — documents are downloaded, converted to markdown, and formatted into user messages
- **Commands**: `/clear`, `/compact`, `/status`
- **WhitelistMiddleware** — optional user ID allow-list

### Tool System

Tools are defined as OpenAI function-calling schemas in [tools.py](src/deeper_bot/tools.py):

| Tool         | Purpose                                                                               |
| ------------ | ------------------------------------------------------------------------------------- |
| `web_search` | DuckDuckGo text search (max 15 results)                                               |
| `web_fetch`  | Fetch page content via httpx + trafilatura extraction. Auto-summarizes if >15K chars. |
| `ask_user`   | Send question to user, block for up to 60 min awaiting reply                          |
| `set_status` | Update research TODO list. Announced to user on first call only.                      |
| `finish`     | Deliver final research report (inline if short, as .md file if long)                  |

All tool arguments are validated via Pydantic models before dispatch.

### SSRF Protection

Web fetching uses a custom `httpx.AsyncHTTPTransport` ([tools.py](src/deeper_bot/tools.py)) that resolves hostnames to IPs at the socket level and blocks:

- Private ranges (10.x, 172.16-31.x, 192.168.x)
- Loopback (127.x, ::1)
- Link-local (169.254.x)
- Reserved and multicast addresses

This eliminates DNS rebinding TOCTOU attacks.

## Code Conventions

### Style

- **ruff** for linting and formatting; line length 120
- Target Python 3.14; use modern syntax (`str | None`, walrus where appropriate)
- Async/await throughout; no synchronous I/O in the hot path
- Type hints on all public functions

### Imports

```python
import asyncio
import logging

from aiogram import Bot

from deeper_bot.config import Settings
```

- Standard library first
- Third-party next
- Internal modules last, absolute imports only

### Error Handling Philosophy

1. **Never leak technical details to users** — agent loop catches all exceptions, sends generic messages
2. **Log with context** — always include `chat_id` in log messages
3. **Never swallow CancelledError** — always re-raise to allow proper asyncio cancellation
4. **Graceful degradation** — retry transient LLM errors, fall back to truncation on summarization failure

### Constants and Limits

Key limits are defined as module-level constants near the top of relevant files:

- `MAX_AGENT_ITERATIONS = 50` (agent.py)
- `MAX_COMPACTION_RETRIES = 2` (agent.py)
- `MAX_TELEGRAM_MESSAGE_LENGTH = 4096` (agent.py)
- `MAX_FETCH_CONTENT_LENGTH = 15_000` (tools.py)
- `MAX_DOWNLOAD_SIZE = 5_000_000` (tools.py)
- `MAX_FILE_CONTENT_LENGTH = 100_000` (converter.py)

## Testing

- **pytest + pytest-asyncio** with `asyncio_mode = auto`
- Tests are organized by module: `test_agent.py`, `test_bot.py`, `test_tools.py`, etc.
- Heavy use of `unittest.mock.AsyncMock` and `MagicMock` for external dependencies (Bot, LLM responses, HTTP)
- SessionStore tests use `tmp_path` for isolated SQLite databases
- Tests verify: error message sanitization, state machine transitions, media group aggregation, SSRF blocking, compaction logic, eviction behavior

Run tests:

```bash
uv run pytest
uv run pytest --cov=src/deeper_bot --cov-report=term-missing
```

## Common Patterns and Pitfalls

### Adding a New Tool

1. Add the OpenAI function schema to `TOOLS` list in [tools.py](src/deeper_bot/tools.py)
2. Create a Pydantic argument model (e.g., `MyToolArgs`) and register it in `_TOOL_MODELS`
3. Implement `async def _my_tool(...)` following existing patterns
4. Add dispatch case in `execute_tool()`
5. Add tests in `test_tools.py`

### Modifying Session Schema

- The `Session` dataclass and SQLite schema must stay in sync
- Add new columns via `ALTER TABLE` in `SessionStore.init()` with duplicate-column error handling (see existing migration pattern)
- Update `_evict_stale()` if new fields affect eviction logic

### Working with asyncio.Future for ask_user

- `set_awaiting_answer()` stores the future and changes state
- `resolve_answer()` sets the result and transitions back to RESEARCHING
- `cancel_pending()` cancels the future (used by /clear)
- Always check `future.done()` before calling `set_result()` or `cancel()`

### Telegram HTML Rendering

- Telegram supports a limited HTML subset: `<b>`, `<i>`, `<u>`, `<s>`, `<code>`, `<pre>`, `<a>`, `<blockquote>`
- The custom `TelegramHTMLRenderer` ([tools.py](src/deeper_bot/tools.py)) converts Markdown to this subset
- Tables are rendered as bulleted lists with bold headers (Telegram has no native table support)
- Images become links
- Always use `markdown_to_telegram_html()` before sending content

### HTTP Client Lifecycle

- The shared `httpx.AsyncClient` is lazily initialized in `_get_http_client()`
- Must call `close_http_client()` on shutdown (registered in `__main__.py` finally block)
- The client uses `_SSRFSafeTransport` — never bypass it for external URLs

### LLM Retry Logic

- `llm_call_with_retry()` ([llm.py](src/deeper_bot/llm.py)) retries on: RateLimitError, ServiceUnavailableError, APIConnectionError, Timeout, InternalServerError
- Delays: 2s, then 4s, then fails
- Always use this wrapper instead of calling `litellm.acompletion()` directly

## Configuration

Environment variables (loaded from `.env`):

| Variable               | Required | Default            | Notes                                |
| ---------------------- | -------- | ------------------ | ------------------------------------ |
| `BOT_TOKEN`            | Yes      | —                  | From @BotFather                      |
| `LLM_BASE_URL`         | Yes      | —                  | OpenAI-compatible endpoint           |
| `LLM_MODEL`            | Yes      | —                  | Primary model identifier             |
| `LLM_API_KEY`          | Yes      | —                  | Direct key or `${ENV_VAR}` reference |
| `LLM_USE_REASONING`    | No       | `true`             | Enable reasoning_effort param        |
| `LLM_REASONING_EFFORT` | No       | `high`             | low/medium/high                      |
| `LLM_UTILITY_MODEL`    | No       | `LLM_MODEL`        | Cheaper model for summarization      |
| `ALLOWED_USERS`        | No       | `[]`               | Comma-separated Telegram user IDs    |
| `DATABASE_PATH`        | No       | `data/sessions.db` | SQLite file path                     |

## Entry Points

```bash
# Development
uv sync --all-extras
uv run -m deeper_bot

# Testing
uv run pytest

# Linting
uv run ruff check .
uv run ruff check --fix .
uv run ruff format .
```
