# Deeper Bot

A Telegram bot for deep research. The bot acts as a ReAct agent that can search the web, analyze sources, and carry out multi-step research sessions directly inside a Telegram chat.

## Features

- **Deep Research Agent** — Multi-turn ReAct loop with tool use (web search, content extraction, asking clarifying questions, final answer delivery)
- **Web Search & Scraping** — DuckDuckGo search with automatic content extraction via Trafilatura
- **Document Uploads** — Attach PDF, DOCX, XLSX, PPTX, code files, and plain-text documents for analysis
- **Session Persistence** — SQLite-backed session storage with context compaction support
- **Access Control** — Optional allow-list for Telegram user IDs
- **LLM-Agnostic** — Powered by LiteLLM; works with OpenAI, Anthropic, local models, or any OpenAI-compatible endpoint
- **Reasoning Support** — Configurable reasoning effort (e.g., OpenAI `o1`/`o3` models)
- **Graceful Degradation** — Automatic retry with exponential backoff, context-window compaction, and cancellation handling

## Tech Stack

| Component           | Library                                                                                                                                                                                                                                                                       |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Telegram API        | [aiogram](https://docs.aiogram.dev/) 3.x                                                                                                                                                                                                                                      |
| LLM Gateway         | [litellm](https://docs.litellm.ai/)                                                                                                                                                                                                                                           |
| Web Search          | [DuckDuckGo Search (ddgs)](https://pypi.org/project/duckduckgo-search/)                                                                                                                                                                                                       |
| Content Extraction  | [trafilatura](https://trafilatura.readthedocs.io/)                                                                                                                                                                                                                            |
| Document Conversion | [pdfplumber](https://github.com/jsvine/pdfplumber) / [pypdf](https://github.com/py-pdf/pypdf) / [python-docx](https://github.com/python-openxml/python-docx) / [openpyxl](https://foss.heptapod.net/openpyxl/openpyxl) / [python-pptx](https://github.com/scanny/python-pptx) |
| HTTP Client         | [httpx](https://www.python-httpx.org/)                                                                                                                                                                                                                                        |
| Configuration       | [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)                                                                                                                                                                                             |
| Database            | [aiosqlite](https://github.com/omnilib/aiosqlite) (SQLite)                                                                                                                                                                                                                    |
| Markdown            | [mistune](https://mistune.lepture.com/)                                                                                                                                                                                                                                       |
| Build Tool          | [uv](https://docs.astral.sh/uv/)                                                                                                                                                                                                                                              |
| Build Backend       | [hatchling](https://hatch.pypa.io/)                                                                                                                                                                                                                                           |
| Linting             | [ruff](https://docs.astral.sh/ruff/)                                                                                                                                                                                                                                          |
| Testing             | [pytest](https://docs.pytest.org/) + [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)                                                                                                                                                                                 |

## Prerequisites

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- An LLM API key (OpenAI, Anthropic, or any compatible provider)

## Local Development

### 1. Clone & Install Dependencies

```bash
git clone <repository-url>
cd deeper-bot
uv sync --all-extras
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and fill in the required values:

```dotenv
# Telegram bot token (from @BotFather)
BOT_TOKEN=your-telegram-bot-token

# LLM configuration
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o
# LLM API key — paste directly or reference an env var: ${OPENAI_API_KEY}
LLM_API_KEY=${OPENAI_API_KEY}
LLM_USE_REASONING=true
LLM_REASONING_EFFORT=high

# Comma-separated list of allowed Telegram user IDs (numeric, not usernames).
# Leave empty to allow all users. Example: ALLOWED_USERS=[123456789,987654321]
ALLOWED_USERS=

# Path to SQLite database file
DATABASE_PATH=data/sessions.db
```

Make sure the referenced environment variable is actually set (not needed if you paste the key directly):

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Run the Bot

```bash
uv run -m deeper_bot
```

Or, if your virtual environment is already activated:

```bash
python -m deeper_bot
```

The bot will start polling Telegram for updates.

### 4. Run Tests

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=src/deeper_bot --cov-report=term-missing
```

### 5. Lint & Format

```bash
uv run ruff check .
uv run ruff check --fix .
uv run ruff format .
```

## Server Deployment

### Option A: Systemd Service (Recommended for Linux VPS)

1. **Create a dedicated user and project directory**

   ```bash
   sudo useradd -r -s /bin/false deeperbot
   sudo mkdir -p /opt/deeper-bot
   sudo chown deeperbot:deeperbot /opt/deeper-bot
   ```

2. **Deploy the code**

   ```bash
   git clone <repository-url> /opt/deeper-bot
   cd /opt/deeper-bot
   sudo -u deeperbot HOME=/opt/deeper-bot uv sync --all-extras --no-dev
   sudo chown -R deeperbot:deeperbot /opt/deeper-bot
   ```

3. **Create the environment file**

   ```bash
   sudo nano /opt/deeper-bot/.env
   ```

   Paste and edit your production configuration (see Local Development section for reference).

4. **Create required directories**

   ```bash
   sudo mkdir -p /opt/deeper-bot/data
   sudo chown deeperbot:deeperbot /opt/deeper-bot/data

   sudo mkdir -p /opt/deeper-bot/.cache
   sudo chown deeperbot:deeperbot /opt/deeper-bot/.cache
   ```

5. **Create a systemd service**

   ```bash
   sudo nano /etc/systemd/system/deeper-bot.service
   ```

   ```ini
   [Unit]
   Description=Deeper Bot Telegram Service
   After=network.target

   [Service]
   Type=simple
   User=deeperbot
   Group=deeperbot
   WorkingDirectory=/opt/deeper-bot
   Environment="PYTHONUNBUFFERED=1"
   Environment="HOME=/opt/deeper-bot"
   Environment="UV_CACHE_DIR=/opt/deeper-bot/.cache"
   EnvironmentFile=/opt/deeper-bot/.env
   ExecStart=/usr/local/bin/uv run -m deeper_bot
   Restart=on-failure
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

   > Adjust the `ExecStart` path to `uv` if it is installed elsewhere (`which uv`).

6. **Enable and start the service**

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now deeper-bot
   sudo systemctl status deeper-bot
   ```

7. **View logs**

   ```bash
   sudo journalctl -u deeper-bot -f
   ```

### Option B: Docker Deployment

Create a `Dockerfile` in the project root:

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "-m", "deeper_bot"]
```

Build and run:

```bash
docker build -t deeper-bot .
docker run -d \
  --name deeper-bot \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  --restart unless-stopped \
  deeper-bot
```

### Option C: Docker Compose

Create a `docker-compose.yml`:

```yaml
services:
  bot:
    build: .
    container_name: deeper-bot
    env_file: .env
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

```bash
docker compose up -d
docker compose logs -f
```

## Removing the Bot from Server

### Systemd Service

```bash
# Stop and disable the service
sudo systemctl stop deeper-bot
sudo systemctl disable deeper-bot

# Remove the service file
sudo rm /etc/systemd/system/deeper-bot.service
sudo systemctl daemon-reload

# Remove user and project directory (optional)
sudo userdel deeperbot
sudo rm -rf /opt/deeper-bot
```

### Docker

```bash
# Stop and remove the container
docker stop deeper-bot
docker rm deeper-bot

# Remove the image (optional)
docker rmi deeper-bot

# Remove data volume (optional)
docker volume prune
```

### Docker Compose

```bash
# Stop and remove containers
docker compose down

# Remove images and volumes (optional)
docker compose down --volumes --rmi all
```

## Bot Commands

| Command    | Description                                        |
| ---------- | -------------------------------------------------- |
| `/clear`   | Clear the current session context                  |
| `/compact` | Compact the conversation context to free up tokens |
| `/status`  | Show current research progress                     |

## Project Structure

```
deeper-bot/
├── src/deeper_bot/
│   ├── __init__.py
│   ├── __main__.py      # Application entry point
│   ├── agent.py         # ReAct agent loop
│   ├── bot.py           # Telegram handlers and middleware
│   ├── compaction.py    # Context compaction logic
│   ├── config.py        # Pydantic settings
│   ├── converter.py     # File-to-markdown conversion
│   ├── llm.py           # LLM client with retry logic
│   ├── prompts.py       # System prompts
│   ├── session.py       # SQLite session store
│   └── tools.py         # Agent tools (search, scrape, etc.)
├── tests/
│   ├── test_agent.py
│   ├── test_bot.py
│   ├── test_compaction.py
│   ├── test_config.py
│   ├── test_session.py
│   └── test_tools.py
├── .env.example
├── pyproject.toml
├── uv.lock
└── README.md
```

## Environment Variables Reference

| Variable               | Required | Default            | Description                                                       |
| ---------------------- | -------- | ------------------ | ----------------------------------------------------------------- |
| `BOT_TOKEN`            | Yes      | —                  | Telegram Bot API token                                            |
| `LLM_BASE_URL`         | Yes      | —                  | Base URL of the LLM API                                           |
| `LLM_MODEL`            | Yes      | —                  | Model identifier (e.g., `gpt-4o`)                                 |
| `LLM_API_KEY`          | Yes      | —                  | LLM API key, or `${ENV_VAR}` to reference an environment variable |
| `LLM_USE_REASONING`    | No       | `true`             | Enable reasoning effort parameter                                 |
| `LLM_REASONING_EFFORT` | No       | `high`             | Reasoning effort level                                            |
| `LLM_UTILITY_MODEL`    | No       | —                  | Fallback model for utility tasks (summarization, compaction)      |
| `ALLOWED_USERS`        | No       | _(empty)_          | Comma-separated numeric Telegram user IDs allowed to use the bot  |
| `DATABASE_PATH`        | No       | `data/sessions.db` | Path to the SQLite database                                       |

## License

MIT
