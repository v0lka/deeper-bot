"""Tests for session recovery on restart."""

import asyncio
from unittest.mock import AsyncMock, patch

from deeper_bot.recovery import RECOVERY_TOOL_CONTENT, recover_sessions, repair_message_history
from deeper_bot.session import SessionState, SessionStore

# ---------------------------------------------------------------------------
# TestRepairMessageHistory
# ---------------------------------------------------------------------------


class TestRepairMessageHistory:
    def test_empty_messages(self):
        messages: list[dict] = []
        repair_message_history(messages)
        assert messages == []

    def test_no_tool_calls_unchanged(self):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        original = [m.copy() for m in messages]
        repair_message_history(messages)
        assert messages == original

    def test_complete_tool_results_unchanged(self):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Search for X"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "web_search", "arguments": '{"query": "X"}'}},
                    {"id": "call_2", "function": {"name": "web_fetch", "arguments": '{"url": "http://x.com"}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Result 1"},
            {"role": "tool", "tool_call_id": "call_2", "content": "Result 2"},
        ]
        original_len = len(messages)
        repair_message_history(messages)
        assert len(messages) == original_len

    def test_missing_single_result_repaired(self):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Search"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "web_search", "arguments": '{"query": "X"}'}},
                ],
            },
            # No tool result for call_1
        ]
        repair_message_history(messages)
        assert len(messages) == 4
        assert messages[3]["role"] == "tool"
        assert messages[3]["tool_call_id"] == "call_1"
        assert messages[3]["content"] == RECOVERY_TOOL_CONTENT

    def test_missing_multiple_results_repaired(self):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Search"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}},
                    {"id": "call_2", "function": {"name": "web_fetch", "arguments": "{}"}},
                    {"id": "call_3", "function": {"name": "set_status", "arguments": "{}"}},
                ],
            },
            # No tool results at all
        ]
        repair_message_history(messages)
        assert len(messages) == 6
        for i, tc_id in enumerate(["call_1", "call_2", "call_3"], start=3):
            assert messages[i]["role"] == "tool"
            assert messages[i]["tool_call_id"] == tc_id
            assert messages[i]["content"] == RECOVERY_TOOL_CONTENT

    def test_partial_results_only_fills_missing(self):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Search"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}},
                    {"id": "call_2", "function": {"name": "web_fetch", "arguments": "{}"}},
                    {"id": "call_3", "function": {"name": "set_status", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Real result"},
            # call_2 and call_3 missing
        ]
        repair_message_history(messages)
        assert len(messages) == 6
        # call_1 result preserved
        assert messages[3]["content"] == "Real result"
        # call_2 and call_3 get synthetic results
        assert messages[4]["tool_call_id"] == "call_2"
        assert messages[4]["content"] == RECOVERY_TOOL_CONTENT
        assert messages[5]["tool_call_id"] == "call_3"
        assert messages[5]["content"] == RECOVERY_TOOL_CONTENT

    def test_idempotent_no_double_repair(self):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Search"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}},
                ],
            },
            # No tool result
        ]
        repair_message_history(messages)
        assert len(messages) == 4
        # Call again — should not add another synthetic result
        repair_message_history(messages)
        assert len(messages) == 4

    def test_non_trailing_incomplete_not_affected(self):
        """Only the LAST assistant message with tool_calls is repaired."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First query"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_old", "function": {"name": "web_search", "arguments": "{}"}},
                ],
            },
            # Missing result for call_old (but it's not the last assistant with tool_calls)
            {"role": "user", "content": "Second query"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_new", "function": {"name": "web_search", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_new", "content": "New result"},
        ]
        original_len = len(messages)
        repair_message_history(messages)
        # Last assistant's tool_calls are complete → no change
        assert len(messages) == original_len


# ---------------------------------------------------------------------------
# TestRecoverSessions
# ---------------------------------------------------------------------------


class TestRecoverSessions:
    async def test_no_interrupted_sessions(self, session_store, bot, settings, bot_state):
        """No interrupted sessions → no agent launches."""
        await recover_sessions(session_store, bot, settings, bot_state)
        assert bot_state.active_tasks == {}

    async def test_single_researching_session_resumed(self, session_store, bot, settings, bot_state):
        """A RESEARCHING session should have its agent loop re-launched."""
        session = await session_store.get_or_create(42)
        session.state = SessionState.RESEARCHING
        session.messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Research topic"},
        ]
        session.research_start_idx = 1
        await session_store.save(session)
        # Evict from memory cache to simulate fresh restart
        session_store._sessions.clear()

        hang = asyncio.Event()

        async def hanging_agent(*args, **kwargs):
            await hang.wait()

        with (
            patch("deeper_bot.recovery.run_agent", side_effect=hanging_agent) as mock_agent,
            patch("deeper_bot.recovery.asyncio.sleep", new_callable=AsyncMock),
        ):
            await recover_sessions(session_store, bot, settings, bot_state)

        mock_agent.assert_called_once()
        call_args = mock_agent.call_args
        assert call_args[0][2] == 42  # chat_id
        assert 42 in bot_state.active_tasks
        hang.set()

    async def test_awaiting_answer_transitions_to_researching(self, session_store, bot, settings, bot_state):
        """AWAITING_ANSWER sessions should transition to RESEARCHING before agent launch."""
        session = await session_store.get_or_create(100)
        session.state = SessionState.AWAITING_ANSWER
        session.messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Query"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_ask", "function": {"name": "ask_user", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_ask", "content": "Waiting for user answer"},
        ]
        session.research_start_idx = 1
        await session_store.save(session)
        session_store._sessions.clear()

        hang = asyncio.Event()

        async def hanging_agent(*args, **kwargs):
            await hang.wait()

        with (
            patch("deeper_bot.recovery.run_agent", side_effect=hanging_agent),
            patch("deeper_bot.recovery.asyncio.sleep", new_callable=AsyncMock),
        ):
            await recover_sessions(session_store, bot, settings, bot_state)

        # Session should have been recovered (task registered)
        assert 100 in bot_state.active_tasks
        hang.set()

    async def test_task_registered_in_bot_state(self, session_store, bot, settings, bot_state):
        """Recovered tasks should be registered in bot_state.active_tasks."""
        session = await session_store.get_or_create(7)
        session.state = SessionState.RESEARCHING
        session.messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Query"},
        ]
        await session_store.save(session)
        session_store._sessions.clear()

        hang = asyncio.Event()

        async def hanging_agent(*args, **kwargs):
            await hang.wait()

        with (
            patch("deeper_bot.recovery.run_agent", side_effect=hanging_agent),
            patch("deeper_bot.recovery.asyncio.sleep", new_callable=AsyncMock),
        ):
            await recover_sessions(session_store, bot, settings, bot_state)

        assert 7 in bot_state.active_tasks
        hang.set()

    async def test_one_failure_doesnt_block_others(self, session_store, bot, settings, bot_state):
        """If one session fails to recover, others should still be processed."""
        # Create two sessions
        for chat_id in (10, 20):
            session = await session_store.get_or_create(chat_id)
            session.state = SessionState.RESEARCHING
            session.messages = [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Query"},
            ]
            await session_store.save(session)
        session_store._sessions.clear()

        call_count = 0

        async def failing_then_ok(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated failure")

        with patch("deeper_bot.recovery.run_agent", side_effect=failing_then_ok):
            await recover_sessions(session_store, bot, settings, bot_state)

        # At least one session should have been attempted after the first failure
        assert call_count == 2

    async def test_empty_session_reset_to_idle(self, session_store, bot, settings, bot_state):
        """Sessions with no meaningful messages should be reset to IDLE."""
        session = await session_store.get_or_create(99)
        session.state = SessionState.RESEARCHING
        session.messages = [{"role": "system", "content": "System"}]  # Only system prompt
        await session_store.save(session)
        session_store._sessions.clear()

        with patch("deeper_bot.recovery.run_agent", new_callable=AsyncMock) as mock_agent:
            await recover_sessions(session_store, bot, settings, bot_state)

        mock_agent.assert_not_called()
        # Session should be IDLE now
        reloaded = await session_store.get_or_create(99)
        assert reloaded.state == SessionState.IDLE

    async def test_stagger_delay_applied(self, session_store, bot, settings, bot_state):
        """There should be a delay between session recovery launches."""
        for chat_id in (1, 2, 3):
            session = await session_store.get_or_create(chat_id)
            session.state = SessionState.RESEARCHING
            session.messages = [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Query"},
            ]
            await session_store.save(session)
        session_store._sessions.clear()

        with (
            patch("deeper_bot.recovery.run_agent", new_callable=AsyncMock),
            patch("deeper_bot.recovery.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            await recover_sessions(session_store, bot, settings, bot_state)

        # sleep should be called for each successfully recovered session
        assert mock_sleep.call_count >= 2  # At least between sessions (not after last)

    async def test_message_repair_called(self, session_store, bot, settings, bot_state):
        """Message history should be repaired before agent launch."""
        session = await session_store.get_or_create(55)
        session.state = SessionState.RESEARCHING
        session.messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Query"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_broken", "function": {"name": "web_search", "arguments": "{}"}}],
            },
            # Missing tool result
        ]
        session.research_start_idx = 1
        await session_store.save(session)
        session_store._sessions.clear()

        with patch("deeper_bot.recovery.run_agent", new_callable=AsyncMock) as mock_agent:
            await recover_sessions(session_store, bot, settings, bot_state)

        # Check that the session passed to run_agent has the repaired message
        call_args = mock_agent.call_args[0]
        recovered_session = call_args[0]
        # Should have the synthetic tool result appended
        tool_msgs = [m for m in recovered_session.messages if m.get("role") == "tool"]
        assert any(m.get("content") == RECOVERY_TOOL_CONTENT for m in tool_msgs)


# ---------------------------------------------------------------------------
# TestRecoveryIntegration
# ---------------------------------------------------------------------------


class TestRecoveryIntegration:
    async def test_full_recovery_cycle(self, tmp_path, bot, settings, bot_state):
        """End-to-end: save interrupted session, simulate restart, verify recovery."""
        db_path = str(tmp_path / "recovery_test.db")

        # Phase 1: simulate a running bot that crashes
        store1 = SessionStore(db_path)
        await store1.init()

        session = await store1.get_or_create(42)
        session.state = SessionState.RESEARCHING
        session.messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Research AI safety"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "function": {"name": "web_search", "arguments": '{"query": "AI"}'}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Search results..."},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_2", "function": {"name": "web_fetch", "arguments": '{"url": "http://x.com"}'}},
                ],
            },
            # Crash here — call_2 result missing
        ]
        session.research_start_idx = 1
        session.todo_list = "- [ ] Search web\n- [ ] Analyze results"
        await store1.save(session)
        await store1.close()

        # Phase 2: simulate fresh restart
        store2 = SessionStore(db_path)
        await store2.init()

        # Verify interrupted sessions are detected
        interrupted = await store2.get_interrupted_chat_ids()
        assert 42 in interrupted

        # Phase 3: run recovery
        hang = asyncio.Event()

        async def hanging_agent(*args, **kwargs):
            await hang.wait()

        with (
            patch("deeper_bot.recovery.run_agent", side_effect=hanging_agent) as mock_agent,
            patch("deeper_bot.recovery.asyncio.sleep", new_callable=AsyncMock),
        ):
            await recover_sessions(store2, bot, settings, bot_state)

        # Verify agent was launched
        mock_agent.assert_called_once()
        recovered_session = mock_agent.call_args[0][0]

        # Verify message history was repaired
        last_tool = next(m for m in reversed(recovered_session.messages) if m.get("role") == "tool")
        assert last_tool["tool_call_id"] == "call_2"
        assert last_tool["content"] == RECOVERY_TOOL_CONTENT

        # Verify task registered
        assert 42 in bot_state.active_tasks
        hang.set()

        await store2.close()
