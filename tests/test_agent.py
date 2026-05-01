from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

from deeper_bot.agent import run_agent
from deeper_bot.session import Session, SessionState, SessionStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session():
    s = Session(chat_id=1)
    s.messages = [{"role": "system", "content": "You are helpful."}]
    s.research_start_idx = 1
    return s


@pytest.fixture
def bot():
    b = AsyncMock()
    b.send_message = AsyncMock()
    b.send_chat_action = AsyncMock()
    return b


@pytest.fixture
def settings():
    s = MagicMock()
    s.llm_model = "test-model"
    s.llm_base_url = "http://localhost"
    s.llm_api_key = "test-key"
    s.llm_use_reasoning = False
    return s


@pytest.fixture
async def session_store(tmp_path):
    store = SessionStore(str(tmp_path / "test.db"))
    await store.init()
    yield store
    await store.close()


# ---------------------------------------------------------------------------
# Agent loop tests
# ---------------------------------------------------------------------------


class TestAgentLoop:
    async def test_error_message_sanitized(self, session, bot, settings, session_store):
        """LLM errors should NOT leak details to the user."""
        with patch("deeper_bot.agent.llm_call_with_retry", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("secret-api-key-12345")
            await run_agent(session, bot, 1, settings, session_store)

        # Check that the user received a generic message, not the exception details
        sent_messages = [call.args[1] for call in bot.send_message.call_args_list if len(call.args) >= 2]
        for msg in sent_messages:
            assert "secret-api-key" not in msg
        assert session.state == SessionState.IDLE

    async def test_context_window_triggers_compaction(self, session, bot, settings, session_store):
        call_count = 0

        async def mock_llm(kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise litellm.ContextWindowExceededError("too long", "test", "test")
            # Second call: return a text response
            msg = MagicMock()
            msg.tool_calls = None
            msg.content = "Here is the answer."
            msg.model_dump.return_value = {"role": "assistant", "content": "Here is the answer."}
            response = MagicMock()
            response.choices = [MagicMock(message=msg)]
            return response

        with (
            patch("deeper_bot.agent.llm_call_with_retry", side_effect=mock_llm),
            patch("deeper_bot.agent.compact_context", new_callable=AsyncMock) as mock_compact,
        ):
            await run_agent(session, bot, 1, settings, session_store)

        mock_compact.assert_called_once()
        assert call_count == 2

    async def test_research_start_idx_updated_on_text_response(self, session, bot, settings, session_store):
        """When model responds with text (no tool calls), research_start_idx should be updated."""
        session.messages.append({"role": "user", "content": "Hello"})

        msg = MagicMock()
        msg.tool_calls = None
        msg.content = "Text response"
        msg.model_dump.return_value = {"role": "assistant", "content": "Text response"}
        response = MagicMock()
        response.choices = [MagicMock(message=msg)]

        with patch("deeper_bot.agent.llm_call_with_retry", new_callable=AsyncMock, return_value=response):
            await run_agent(session, bot, 1, settings, session_store)

        assert session.state == SessionState.IDLE
        # research_start_idx should point past all current messages
        assert session.research_start_idx == len(session.messages)

    async def test_short_text_response_sent_inline(self, session, bot, settings, session_store):
        """Short text responses should be sent as inline HTML messages."""
        session.messages.append({"role": "user", "content": "Hello"})

        msg = MagicMock()
        msg.tool_calls = None
        msg.content = "Short reply"
        msg.model_dump.return_value = {"role": "assistant", "content": "Short reply"}
        response = MagicMock()
        response.choices = [MagicMock(message=msg)]

        with patch("deeper_bot.agent.llm_call_with_retry", new_callable=AsyncMock, return_value=response):
            await run_agent(session, bot, 1, settings, session_store)

        bot.send_message.assert_called()
        bot.send_document.assert_not_called()

    async def test_long_text_response_sent_as_file(self, session, bot, settings, session_store):
        """Text responses exceeding Telegram limit should be sent as a .md file."""
        session.messages.append({"role": "user", "content": "Hello"})

        long_content = "# " + "x" * 5000
        msg = MagicMock()
        msg.tool_calls = None
        msg.content = long_content
        msg.model_dump.return_value = {"role": "assistant", "content": long_content}
        response = MagicMock()
        response.choices = [MagicMock(message=msg)]

        with patch("deeper_bot.agent.llm_call_with_retry", new_callable=AsyncMock, return_value=response):
            await run_agent(session, bot, 1, settings, session_store)

        bot.send_document.assert_called_once()
        doc_arg = bot.send_document.call_args.args[1]
        assert doc_arg.filename == "response.md"
        assert session.state == SessionState.IDLE

    async def test_finish_tool_ends_loop(self, session, bot, settings, session_store):
        session.messages.append({"role": "user", "content": "Research X"})

        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "finish"
        tc.function.arguments = '{"result_markdown": "# Done"}'

        msg = MagicMock()
        msg.tool_calls = [tc]
        msg.content = None
        finish_args = '{"result_markdown": "# Done"}'
        msg.model_dump.return_value = {
            "role": "assistant",
            "tool_calls": [{"id": "call_1", "function": {"name": "finish", "arguments": finish_args}}],
        }
        response = MagicMock()
        response.choices = [MagicMock(message=msg)]

        with patch("deeper_bot.agent.llm_call_with_retry", new_callable=AsyncMock, return_value=response):
            await run_agent(session, bot, 1, settings, session_store)

        assert session.state == SessionState.IDLE
        assert session.research_start_idx == len(session.messages)

    async def test_todo_list_injected_into_llm_messages(self, session, bot, settings, session_store):
        """When session has a todo_list, it should be injected into the LLM call messages."""
        session.messages.append({"role": "user", "content": "Research X"})
        session.todo_list = "- [ ] Step 1\n- [ ] Step 2"

        msg = MagicMock()
        msg.tool_calls = None
        msg.content = "Response"
        msg.model_dump.return_value = {"role": "assistant", "content": "Response"}
        response = MagicMock()
        response.choices = [MagicMock(message=msg)]

        captured_kwargs = {}

        async def capture_llm(kwargs):
            captured_kwargs.update(kwargs)
            return response

        with patch("deeper_bot.agent.llm_call_with_retry", side_effect=capture_llm):
            await run_agent(session, bot, 1, settings, session_store)

        # The LLM should have received the ephemeral status message
        llm_messages = captured_kwargs["messages"]
        status_msgs = [m for m in llm_messages if m.get("content", "").startswith("## Current Research Progress")]
        assert len(status_msgs) == 1
        assert "Step 1" in status_msgs[0]["content"]

        # But session.messages should NOT contain the ephemeral message
        for m in session.messages:
            assert "Current Research Progress" not in m.get("content", "")

    async def test_todo_list_not_injected_when_none(self, session, bot, settings, session_store):
        """When session has no todo_list, no extra message should be injected."""
        session.messages.append({"role": "user", "content": "Hello"})
        assert session.todo_list is None

        msg = MagicMock()
        msg.tool_calls = None
        msg.content = "Response"
        msg.model_dump.return_value = {"role": "assistant", "content": "Response"}
        response = MagicMock()
        response.choices = [MagicMock(message=msg)]

        captured_kwargs = {}

        async def capture_llm(kwargs):
            captured_kwargs.update(kwargs)
            return response

        with patch("deeper_bot.agent.llm_call_with_retry", side_effect=capture_llm):
            await run_agent(session, bot, 1, settings, session_store)

        llm_messages = captured_kwargs["messages"]
        status_msgs = [m for m in llm_messages if m.get("content", "").startswith("## Current Research Progress")]
        assert len(status_msgs) == 0

    async def test_clear_status_on_completion(self, session, bot, settings, session_store):
        """After run_agent completes, todo_list should be cleared."""
        session.messages.append({"role": "user", "content": "Hello"})
        session.todo_list = "- [x] Done"
        session._status_announced = True

        msg = MagicMock()
        msg.tool_calls = None
        msg.content = "Response"
        msg.model_dump.return_value = {"role": "assistant", "content": "Response"}
        response = MagicMock()
        response.choices = [MagicMock(message=msg)]

        with patch("deeper_bot.agent.llm_call_with_retry", new_callable=AsyncMock, return_value=response):
            await run_agent(session, bot, 1, settings, session_store)

        assert session.todo_list is None
        assert session._status_announced is False


class TestTelegramForbiddenHandling:
    async def test_telegram_forbidden_handled_gracefully(self, session, bot, settings, session_store):
        """TelegramForbiddenError should be handled gracefully without sending error messages."""
        from aiogram.exceptions import TelegramForbiddenError

        with patch("deeper_bot.agent.llm_call_with_retry", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = TelegramForbiddenError(
                method=MagicMock(),
                message="Forbidden: bot was blocked by the user",
            )
            await run_agent(session, bot, 1, settings, session_store)

        # Should NOT try to send the generic error message
        for call in bot.send_message.call_args_list:
            assert "unexpected error" not in call.args[1].lower()
        assert session.state == SessionState.IDLE

    async def test_keep_typing_handles_telegram_forbidden(self):
        """_keep_typing should stop gracefully when bot is blocked."""
        from aiogram.exceptions import TelegramForbiddenError
        from deeper_bot.agent import _keep_typing

        bot = AsyncMock()
        bot.send_chat_action = AsyncMock(
            side_effect=TelegramForbiddenError(
                method=MagicMock(),
                message="Forbidden: bot was blocked by the user",
            )
        )

        await _keep_typing(bot, 1)
        bot.send_chat_action.assert_called_once()
