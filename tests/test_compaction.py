from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deeper_bot.compaction import _render_messages_for_summary, compact_context
from deeper_bot.session import SUMMARY_PREFIX, Session


@pytest.fixture
def settings():
    s = MagicMock()
    s.llm_model = "test-model"
    s.llm_base_url = "http://localhost"
    s.llm_api_key = "test-key"
    return s


# ---------------------------------------------------------------------------
# _render_messages_for_summary tests
# ---------------------------------------------------------------------------


class TestRenderMessagesForSummary:
    def test_skips_system_messages(self):
        messages = [{"role": "system", "content": "sys prompt"}]
        result = _render_messages_for_summary(messages)
        assert result == ""

    def test_renders_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = _render_messages_for_summary(messages)
        assert "User: Hello" in result

    def test_renders_assistant_message(self):
        messages = [{"role": "assistant", "content": "Hi there"}]
        result = _render_messages_for_summary(messages)
        assert "Assistant: Hi there" in result

    def test_renders_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "web_search", "arguments": '{"query": "test"}'}}],
            }
        ]
        result = _render_messages_for_summary(messages)
        assert "web_search" in result

    def test_truncates_long_tool_results(self):
        messages = [{"role": "tool", "content": "x" * 1000}]
        result = _render_messages_for_summary(messages)
        assert "..." in result
        assert len(result) < 1000


# ---------------------------------------------------------------------------
# compact_context tests
# ---------------------------------------------------------------------------


class TestCompactContext:
    async def test_nothing_to_compact(self, settings):
        session = Session(chat_id=1)
        session.messages = [{"role": "system", "content": "sys"}]
        session.research_start_idx = 1
        await compact_context(session, settings)
        # Should be unchanged
        assert len(session.messages) == 1

    async def test_compacts_raw_messages(self, settings):
        session = Session(chat_id=1)
        session.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Answer 2"},
            # current research starts here
            {"role": "user", "content": "Current question"},
        ]
        session.research_start_idx = 5

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary of Q1/A1 and Q2/A2"

        with patch("deeper_bot.compaction.llm_call_with_retry", new_callable=AsyncMock, return_value=mock_response):
            await compact_context(session, settings)

        # Should have: system + summary + current question
        assert len(session.messages) == 3
        assert session.messages[0]["role"] == "system"
        assert session.messages[1]["content"].startswith(SUMMARY_PREFIX)
        assert "Summary of Q1/A1" in session.messages[1]["content"]
        assert session.messages[2]["content"] == "Current question"
        assert session.research_start_idx == 2

    async def test_fallback_on_llm_failure(self, settings):
        session = Session(chat_id=1)
        session.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Current"},
        ]
        session.research_start_idx = 5

        with patch(
            "deeper_bot.compaction.llm_call_with_retry",
            new_callable=AsyncMock,
            side_effect=Exception("LLM down"),
        ):
            await compact_context(session, settings)

        # Should still compact, using fallback truncation
        assert len(session.messages) == 3
        assert session.messages[0]["role"] == "system"
        assert session.messages[1]["content"].startswith(SUMMARY_PREFIX)
        assert session.research_start_idx == 2

    async def test_removes_old_summaries_only(self, settings):
        session = Session(chat_id=1)
        session.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": SUMMARY_PREFIX + "old summary"},
            {"role": "user", "content": "Current"},
        ]
        session.research_start_idx = 2

        await compact_context(session, settings)

        # Old summary removed, only system + current remain
        assert len(session.messages) == 2
        assert session.messages[0]["role"] == "system"
        assert session.messages[1]["content"] == "Current"
        assert session.research_start_idx == 1

    async def test_double_compact_removes_summary(self, settings):
        """A second compact with no new raw messages should remove the previous summary.

        After compacting raw messages into a summary, a second compact with no new raw
        messages should remove the summary.
        """
        session = Session(chat_id=1)
        session.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Answer 2"},
        ]
        session.research_start_idx = 5

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary of Q1/A1 and Q2/A2"

        with patch("deeper_bot.compaction.llm_call_with_retry", new_callable=AsyncMock, return_value=mock_response):
            await compact_context(session, settings)

        # After first compact: system + summary
        assert len(session.messages) == 2
        assert session.messages[1]["content"].startswith(SUMMARY_PREFIX)
        assert session.research_start_idx == 2

        # Second compact should remove the summary since there are no raw messages
        await compact_context(session, settings)

        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "system"
        assert session.research_start_idx == 1
