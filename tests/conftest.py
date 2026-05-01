"""Shared test fixtures for deeper-bot tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from deeper_bot.bot import BotState
from deeper_bot.session import Session, SessionStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings():
    """Mock Settings with common defaults for testing."""
    s = MagicMock()
    s.llm_model = "test-model"
    s.llm_base_url = "http://localhost"
    s.llm_api_key = "test-key"
    s.llm_use_reasoning = False
    s.llm_reasoning_effort = "high"
    s.resolved_llm_api_key = "test-key"
    s.resolved_utility_model = "test-model"
    s.allowed_users = []
    return s


@pytest.fixture
def bot():
    """AsyncMock Bot with common Telegram methods."""
    b = AsyncMock()
    b.send_message = AsyncMock()
    b.send_chat_action = AsyncMock()
    b.send_document = AsyncMock()
    b.get_file = AsyncMock()
    b.download_file = AsyncMock()
    return b


@pytest.fixture
async def session_store(tmp_path):
    """Async SessionStore backed by a temporary SQLite database."""
    s = SessionStore(str(tmp_path / "test.db"))
    await s.init()
    yield s
    await s.close()


@pytest.fixture
def session():
    """Session pre-loaded with a system message."""
    s = Session(chat_id=1)
    s.messages = [{"role": "system", "content": "You are helpful."}]
    s.research_start_idx = 1
    return s


# Alias for test files that use "store" instead of "session_store"
@pytest.fixture
async def store(session_store):
    """Alias for session_store fixture."""
    return session_store


@pytest.fixture
def bot_state():
    """Fresh BotState instance for each test."""
    return BotState()
