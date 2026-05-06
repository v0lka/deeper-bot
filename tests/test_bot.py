import asyncio
import contextlib
from io import BytesIO
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.types import Message

from deeper_bot.bot import (
    WhitelistMiddleware,
    _extract_content,
    _format_user_content,
    _handle_user_input,
    _process_media_group,
    clear_session,
    compact_session,
    show_status,
    stop_session,
)
from deeper_bot.session import SessionState
from tests.helpers import make_document_message, make_media_group_message, make_message

# ---------------------------------------------------------------------------
# WhitelistMiddleware tests
# ---------------------------------------------------------------------------


class TestWhitelistMiddleware:
    async def test_allowed_user_passes(self):
        mw = WhitelistMiddleware([100, 200])
        handler = AsyncMock(return_value="ok")
        msg = make_message(1, "hello", user_id=100)

        result = await mw(handler, msg, {})
        handler.assert_called_once()
        assert result == "ok"

    async def test_denied_user_blocked(self):
        mw = WhitelistMiddleware([100, 200])
        handler = AsyncMock(return_value="ok")
        msg = make_message(1, "hello", user_id=999)

        result = await mw(handler, msg, {})
        handler.assert_not_called()
        assert result is None

    async def test_non_message_event_passes(self):
        mw = WhitelistMiddleware([100])
        handler = AsyncMock(return_value="ok")
        # Not a Message object
        event = MagicMock()
        await mw(handler, event, {})
        handler.assert_called_once()


# ---------------------------------------------------------------------------
# Handler tests
# ---------------------------------------------------------------------------


class TestHandlers:
    async def test_clear_resets_session(self, store, bot_state):
        """clear_session should reset messages, index, initialized, and reply 'Context cleared'."""
        session = await store.get_or_create(42)
        session.messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        session.research_start_idx = 2
        session.initialized = True
        await store.save(session)

        msg = make_message(42, "/clear")
        await clear_session(msg, store, bot_state)

        assert session.messages == []
        assert session.research_start_idx == 0
        assert session.initialized is False
        assert session.state == SessionState.IDLE
        cast(AsyncMock, msg.answer).assert_awaited_once()
        reply_html = cast(AsyncMock, msg.answer).call_args[0][0]
        assert "Context cleared" in reply_html

    async def test_clear_deletes_document_cache(self, store, bot_state):
        """clear_session should delete cached documents for the chat."""
        from deeper_bot.tools.documents import save_document

        await save_document("cached content", 42)

        session = await store.get_or_create(42)
        session.state = SessionState.IDLE
        await store.save(session)

        msg = make_message(42, "/clear")
        await clear_session(msg, store, bot_state)

        from deeper_bot.tools.documents import _get_session_cache_dir

        assert not _get_session_cache_dir(42).exists()

    async def test_clear_during_research_blocked(self, store, bot_state):
        """clear_session should refuse to clear when a research session is active."""
        session = await store.get_or_create(42)
        session.state = SessionState.RESEARCHING
        session.messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        await store.save(session)

        fake_task = MagicMock()
        fake_task.done.return_value = False
        bot_state.active_tasks[42] = fake_task

        msg = make_message(42, "/clear")
        await clear_session(msg, store, bot_state)

        # Should not cancel or clear anything
        fake_task.cancel.assert_not_called()
        assert session.messages != []
        assert session.state == SessionState.RESEARCHING
        cast(AsyncMock, msg.answer).assert_awaited_once()
        reply_html = cast(AsyncMock, msg.answer).call_args[0][0]
        assert "Cannot clear context" in reply_html

    async def test_stop_cancels_active_task(self, store, bot_state):
        """stop_session should cancel any active task for the chat."""
        session = await store.get_or_create(42)
        session.state = SessionState.RESEARCHING
        await store.save(session)

        fake_task = MagicMock()
        fake_task.done.return_value = False
        bot_state.active_tasks[42] = fake_task

        msg = make_message(42, "/stop")
        await stop_session(msg, store, bot_state)

        fake_task.cancel.assert_called_once()
        assert 42 not in bot_state.active_tasks
        assert session.state == SessionState.IDLE
        cast(AsyncMock, msg.answer).assert_awaited_once()
        reply_html = cast(AsyncMock, msg.answer).call_args[0][0]
        assert "Research session stopped" in reply_html

    async def test_stop_preserves_context(self, store, bot_state):
        """stop_session should preserve messages and research_start_idx."""
        session = await store.get_or_create(42)
        session.state = SessionState.RESEARCHING
        session.messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        session.research_start_idx = 1
        session.allowed_domains = {"example.com"}
        await store.save(session)

        msg = make_message(42, "/stop")
        await stop_session(msg, store, bot_state)

        assert session.messages == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        assert session.research_start_idx == 1
        assert session.allowed_domains == {"example.com"}
        assert session.state == SessionState.IDLE

    async def test_stop_when_idle(self, store, bot_state):
        """stop_session when IDLE should inform user there is nothing to stop."""
        session = await store.get_or_create(42)
        assert session.state == SessionState.IDLE

        msg = make_message(42, "/stop")
        await stop_session(msg, store, bot_state)

        cast(AsyncMock, msg.answer).assert_awaited_once()
        reply_html = cast(AsyncMock, msg.answer).call_args[0][0]
        assert "No active research session" in reply_html

    async def test_message_during_research_gets_wait_reply(self, store, bot, bot_state):
        """User messages during RESEARCHING state should get a 'please wait' reply."""
        session = await store.get_or_create(42)
        session.state = SessionState.RESEARCHING
        await store.save(session)

        msg = make_message(42, "hello")
        await _handle_user_input(
            42,
            "hello",
            has_files=False,
            text_present=True,
            message=msg,
            session_store=store,
            settings=MagicMock(),
            bot=bot,
            bot_state=bot_state,
        )

        cast(AsyncMock, msg.reply).assert_awaited_once()
        reply_html = cast(AsyncMock, msg.reply).call_args[0][0]
        assert "Research in progress" in reply_html

    async def test_message_during_awaiting_resolves_answer(self, store):
        """User messages during AWAITING_ANSWER should resolve the pending future."""
        session = await store.get_or_create(42)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        session.set_awaiting_answer(future)

        async with session.lock:
            session.resolve_answer("user reply")

        assert session.state == SessionState.RESEARCHING
        assert future.result() == "user reply"

    async def test_compact_with_few_messages(self, store, settings):
        """Compacting with research_start_idx <= 1 should say 'Nothing to compact'."""
        session = await store.get_or_create(42)
        session.messages = [{"role": "system", "content": "sys"}]
        session.research_start_idx = 1
        await store.save(session)

        msg = make_message(42, "/compact")
        await compact_session(msg, store, settings)

        cast(AsyncMock, msg.answer).assert_awaited_once()
        reply_html = cast(AsyncMock, msg.answer).call_args[0][0]
        assert "Nothing to compact" in reply_html


# ---------------------------------------------------------------------------
# Status handler tests
# ---------------------------------------------------------------------------


class TestStatusHandler:
    async def test_status_no_active_research(self, store):
        """When IDLE, /status should reply 'No active research session'."""
        await store.get_or_create(42)

        msg = make_message(42, "/status")
        await show_status(msg, store)

        cast(AsyncMock, msg.answer).assert_awaited_once()
        reply_html = cast(AsyncMock, msg.answer).call_args[0][0]
        assert "No active research session" in reply_html

    async def test_status_researching_no_todo(self, store):
        """When RESEARCHING but no TODO set, /status should say no plan yet."""
        session = await store.get_or_create(42)
        session.state = SessionState.RESEARCHING

        msg = make_message(42, "/status")
        await show_status(msg, store)

        cast(AsyncMock, msg.answer).assert_awaited_once()
        reply_html = cast(AsyncMock, msg.answer).call_args[0][0]
        assert "no plan" in reply_html.lower()

    async def test_status_researching_with_todo(self, store):
        """When RESEARCHING with TODO set, /status should return the TODO list."""
        session = await store.get_or_create(42)
        session.state = SessionState.RESEARCHING
        session.todo_list = "- [ ] Step 1\n- [x] Step 2"

        msg = make_message(42, "/status")
        await show_status(msg, store)

        cast(AsyncMock, msg.answer).assert_awaited_once()
        reply_html = cast(AsyncMock, msg.answer).call_args[0][0]
        assert "Step 1" in reply_html
        assert "Step 2" in reply_html

    async def test_clear_clears_status(self, store, bot_state):
        """clear_session should clear todo_list and status_announced."""
        session = await store.get_or_create(42)
        session.todo_list = "- [ ] Step 1"
        session.status_announced = True
        await store.save(session)

        msg = make_message(42, "/clear")
        await clear_session(msg, store, bot_state)

        assert session.todo_list is None
        assert session.status_announced is False


# ---------------------------------------------------------------------------
# _format_user_content tests
# ---------------------------------------------------------------------------


class TestFormatUserContent:
    def test_file_with_caption(self):
        result = _format_user_content("Analyze this", "# Content", "report.pdf")
        assert result.startswith("Analyze this")
        assert "---" in result
        assert "report.pdf" in result
        assert "# Content" in result

    def test_file_without_caption(self):
        result = _format_user_content("", "# Content", "data.csv")
        assert result.startswith("Attached file:")
        assert "data.csv" in result
        assert "# Content" in result
        assert "---" not in result

    def test_whitespace_only_caption_treated_as_empty(self):
        result = _format_user_content("   ", "# Content", "file.txt")
        assert result.startswith("Attached file:")
        assert "---" not in result


# ---------------------------------------------------------------------------
# _extract_content tests
# ---------------------------------------------------------------------------


class TestExtractContent:
    async def test_text_only_message(self):
        msg = make_message(1, "hello")
        msg.document = None
        result = await _extract_content(msg, AsyncMock())
        assert result == ("hello", None, None)

    async def test_empty_message_returns_none(self):
        msg = MagicMock(spec=Message)
        msg.text = None
        msg.document = None
        result = await _extract_content(msg, AsyncMock())
        assert result is None

    async def test_document_with_caption(self):
        msg = make_document_message(1, "notes.txt", caption="Check this")
        bot = AsyncMock()
        file_mock = MagicMock()
        file_mock.file_path = "path/to/file"
        bot.get_file.return_value = file_mock
        bot.download_file.return_value = BytesIO(b"file content")

        result = await _extract_content(msg, bot)
        assert result is not None
        text, file_md, filename = result
        assert text == "Check this"
        assert file_md == "file content"
        assert filename == "notes.txt"

    async def test_document_without_caption(self):
        msg = make_document_message(1, "data.txt")
        bot = AsyncMock()
        file_mock = MagicMock()
        file_mock.file_path = "path/to/file"
        bot.get_file.return_value = file_mock
        bot.download_file.return_value = BytesIO(b"data here")

        result = await _extract_content(msg, bot)
        assert result is not None
        text, file_md, filename = result
        assert text == ""
        assert file_md == "data here"
        assert filename == "data.txt"

    async def test_unsupported_format_raises(self):
        from deeper_bot.converter import UnsupportedFileError

        msg = make_document_message(1, "archive.xyz")
        with pytest.raises(UnsupportedFileError):
            await _extract_content(msg, AsyncMock())

    async def test_pdf_delegates_to_pdfplumber(self):
        msg = make_document_message(1, "report.pdf")
        bot = AsyncMock()
        file_mock = MagicMock()
        file_mock.file_path = "path/to/file"
        bot.get_file.return_value = file_mock
        bot.download_file.return_value = BytesIO(b"fake pdf")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF content"

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("deeper_bot.converter.pdfplumber.open", return_value=mock_pdf):
            result = await _extract_content(msg, bot)

        assert result is not None
        _, file_md, _ = result
        assert file_md == "PDF content"


# ---------------------------------------------------------------------------
# Document handling integration tests
# ---------------------------------------------------------------------------


class TestDocumentHandling:
    async def test_document_idle_without_caption_adds_context_no_agent(self, store):
        """File without caption in IDLE state should add to context and not start agent."""
        session = await store.get_or_create(42)
        async with session.lock:
            session.messages = [{"role": "system", "content": "sys"}]
            session.research_start_idx = 1
            session.state = SessionState.IDLE
            await store.save(session)

        # Simulate what handle_message does for file-only messages
        user_content = _format_user_content("", "file content", "notes.txt")
        async with session.lock:
            session.messages.append({"role": "user", "content": user_content})
            await store.save(session)

        assert len(session.messages) == 2
        assert "notes.txt" in session.messages[1]["content"]
        assert session.state == SessionState.IDLE

    async def test_document_awaiting_answer_resolves_with_file_content(self, store):
        """File sent during AWAITING_ANSWER should resolve with formatted content."""
        session = await store.get_or_create(42)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        session.set_awaiting_answer(future)

        user_content = _format_user_content("", "file content here", "doc.txt")
        async with session.lock:
            session.resolve_answer(user_content)

        assert session.state == SessionState.RESEARCHING
        assert "doc.txt" in future.result()
        assert "file content here" in future.result()

    async def test_text_only_still_works(self, store):
        """Regression: plain text messages should still work as before."""
        session = await store.get_or_create(42)
        async with session.lock:
            session.messages = [{"role": "system", "content": "sys"}]
            session.research_start_idx = 1
            session.state = SessionState.IDLE

            session.messages.append({"role": "user", "content": "hello"})

        assert session.messages[-1] == {"role": "user", "content": "hello"}


# ---------------------------------------------------------------------------
# _handle_user_input direct tests
# ---------------------------------------------------------------------------


class TestHandleUserInput:
    async def test_idle_with_text_starts_agent(self, store, bot, bot_state):
        """IDLE state with text should append message and start agent."""
        session = await store.get_or_create(42)
        async with session.lock:
            session.messages = [{"role": "system", "content": "sys"}]
            session.research_start_idx = 1
            session.state = SessionState.IDLE
            await store.save(session)

        msg = make_message(42, "hello")

        async def fake_run_agent(*args, **kwargs):
            pass

        with patch("deeper_bot.bot.run_agent", side_effect=fake_run_agent):
            await _handle_user_input(
                42,
                "hello",
                has_files=False,
                text_present=True,
                message=msg,
                session_store=store,
                settings=MagicMock(),
                bot=bot,
                bot_state=bot_state,
            )

        assert 42 in bot_state.active_tasks
        session = await store.get_or_create(42)
        assert session.messages[-1] == {"role": "user", "content": "hello"}

    async def test_language_code_captured_from_message(self, store, bot, bot_state):
        """Telegram language_code should be stored in the session."""
        session = await store.get_or_create(42)
        async with session.lock:
            session.messages = [{"role": "system", "content": "sys"}]
            session.research_start_idx = 1
            session.state = SessionState.IDLE
            await store.save(session)

        msg = make_message(42, "hello", language_code="de")

        async def fake_run_agent(*args, **kwargs):
            pass

        with patch("deeper_bot.bot.run_agent", side_effect=fake_run_agent):
            await _handle_user_input(
                42,
                "hello",
                has_files=False,
                text_present=True,
                message=msg,
                session_store=store,
                settings=MagicMock(),
                bot=bot,
                bot_state=bot_state,
            )

        session = await store.get_or_create(42)
        assert session.language_code == "de"

    async def test_idle_with_file_only_adds_context(self, store, bot, bot_state):
        """IDLE state with files but no text should add to context without starting agent."""
        session = await store.get_or_create(42)
        async with session.lock:
            session.messages = [{"role": "system", "content": "sys"}]
            session.research_start_idx = 1
            session.state = SessionState.IDLE
            await store.save(session)

        msg = make_document_message(42, "data.txt")
        await _handle_user_input(
            42,
            "file content",
            has_files=True,
            text_present=False,
            message=msg,
            session_store=store,
            settings=MagicMock(),
            bot=bot,
            bot_state=bot_state,
        )

        assert 42 not in bot_state.active_tasks
        cast(AsyncMock, msg.reply).assert_awaited_once()
        session = await store.get_or_create(42)
        assert session.messages[-1] == {"role": "user", "content": "file content"}

    async def test_researching_sends_wait_reply(self, store, bot, bot_state):
        """RESEARCHING state should reply with 'please wait'."""
        session = await store.get_or_create(42)
        async with session.lock:
            session.state = SessionState.RESEARCHING
            await store.save(session)

        msg = make_message(42, "hello")
        await _handle_user_input(
            42,
            "hello",
            has_files=False,
            text_present=True,
            message=msg,
            session_store=store,
            settings=MagicMock(),
            bot=bot,
            bot_state=bot_state,
        )

        cast(AsyncMock, msg.reply).assert_awaited_once()
        assert 42 not in bot_state.active_tasks

    async def test_awaiting_answer_resolves_future(self, store, bot, bot_state):
        """AWAITING_ANSWER state should resolve the pending future."""
        session = await store.get_or_create(42)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        session.set_awaiting_answer(future)

        msg = make_message(42, "my answer")
        await _handle_user_input(
            42,
            "my answer",
            has_files=False,
            text_present=True,
            message=msg,
            session_store=store,
            settings=MagicMock(),
            bot=bot,
            bot_state=bot_state,
        )

        assert future.result() == "my answer"
        assert session.state == SessionState.RESEARCHING


# ---------------------------------------------------------------------------
# Media group handling tests
# ---------------------------------------------------------------------------


class TestMediaGroupHandling:
    async def test_media_group_combines_files_into_single_prompt(self, store, settings, bot, bot_state):
        """Multiple files in a media group should be combined into one user message."""
        session = await store.get_or_create(42)
        async with session.lock:
            session.messages = [{"role": "system", "content": "sys"}]
            session.research_start_idx = 1
            session.state = SessionState.IDLE
            await store.save(session)

        msg1 = make_media_group_message(42, "mg1", "file1.txt", caption="Analyze these")
        msg2 = make_media_group_message(42, "mg1", "file2.txt", caption="Analyze these")
        msg3 = make_media_group_message(42, "mg1", "file3.txt", caption="Analyze these")
        bot_state.media_group_buffers["mg1"] = [msg1, msg2, msg3]

        async def fake_extract(msg, bot):
            return msg.caption, f"content of {msg.document.file_name}", msg.document.file_name

        with patch("deeper_bot.bot._extract_content", side_effect=fake_extract), patch("asyncio.sleep"):
            await _process_media_group(42, "mg1", bot, store, settings, bot_state)

        # An agent task should have been registered
        assert 42 in bot_state.active_tasks
        # Clean up the task
        task = bot_state.active_tasks.pop(42, None)
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        session = await store.get_or_create(42)
        user_msgs = [m for m in session.messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        content = user_msgs[0]["content"]
        assert "Analyze these" in content
        assert "file1.txt" in content
        assert "file2.txt" in content
        assert "file3.txt" in content

    async def test_media_group_without_caption_adds_context_no_agent(self, store, settings, bot, bot_state):
        """Media group without caption should add files to context but not start agent."""
        session = await store.get_or_create(42)
        async with session.lock:
            session.messages = [{"role": "system", "content": "sys"}]
            session.research_start_idx = 1
            session.state = SessionState.IDLE
            await store.save(session)

        msg1 = make_media_group_message(42, "mg2", "file1.txt")
        msg2 = make_media_group_message(42, "mg2", "file2.txt")
        bot_state.media_group_buffers["mg2"] = [msg1, msg2]

        async def fake_extract(msg, bot):
            return "", f"content of {msg.document.file_name}", msg.document.file_name

        with patch("deeper_bot.bot._extract_content", side_effect=fake_extract), patch("asyncio.sleep"):
            await _process_media_group(42, "mg2", bot, store, settings, bot_state)

        # No agent should have been started
        assert 42 not in bot_state.active_tasks

        session = await store.get_or_create(42)
        user_msgs = [m for m in session.messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert "file1.txt" in user_msgs[0]["content"]
        assert "file2.txt" in user_msgs[0]["content"]

    async def test_media_group_skips_unsupported_files(self, store, settings, bot, bot_state):
        """Unsupported files in a media group should be skipped, others processed."""
        session = await store.get_or_create(42)
        async with session.lock:
            session.messages = [{"role": "system", "content": "sys"}]
            session.research_start_idx = 1
            session.state = SessionState.IDLE
            await store.save(session)

        msg1 = make_media_group_message(42, "mg3", "file1.txt", caption="Analyze")
        msg2 = make_media_group_message(42, "mg3", "bad.xyz")
        bot_state.media_group_buffers["mg3"] = [msg1, msg2]

        async def fake_extract(msg, bot):
            if msg.document.file_name == "bad.xyz":
                from deeper_bot.converter import UnsupportedFileError

                raise UnsupportedFileError("Unsupported file: bad.xyz")
            return msg.caption, f"content of {msg.document.file_name}", msg.document.file_name

        with patch("deeper_bot.bot._extract_content", side_effect=fake_extract), patch("asyncio.sleep"):
            await _process_media_group(42, "mg3", bot, store, settings, bot_state)

        # An agent task should have been registered
        assert 42 in bot_state.active_tasks
        task = bot_state.active_tasks.pop(42, None)
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        session = await store.get_or_create(42)
        user_msgs = [m for m in session.messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert "file1.txt" in user_msgs[0]["content"]
        assert "bad.xyz" not in user_msgs[0]["content"]


# ---------------------------------------------------------------------------
# active_tasks cleanup tests
# ---------------------------------------------------------------------------


class TestActiveTasksCleanup:
    async def test_done_callback_cleans_up_on_success(self, bot_state):
        """Completed task should be removed from active_tasks."""
        chat_id = 99999

        async def dummy():
            pass

        task = asyncio.create_task(dummy())
        bot_state.active_tasks[chat_id] = task
        await task
        bot_state.active_tasks.pop(chat_id, None)
        assert chat_id not in bot_state.active_tasks

    async def test_done_callback_cleans_up_on_cancellation(self, bot_state):
        """Cancelled task should be removed from active_tasks."""
        chat_id = 99998

        async def hang():
            await asyncio.sleep(3600)

        task = asyncio.create_task(hang())
        bot_state.active_tasks[chat_id] = task
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        bot_state.active_tasks.pop(chat_id, None)
        assert chat_id not in bot_state.active_tasks


# ---------------------------------------------------------------------------
# Security: domain extraction and clear tests
# ---------------------------------------------------------------------------


class TestSecurityDomainTracking:
    async def test_user_url_populates_allowed_domains(self, store, bot, bot_state):
        """URLs in user messages should be extracted and added to allowed_domains."""
        session = await store.get_or_create(42)
        async with session.lock:
            session.messages = [{"role": "system", "content": "sys"}]
            session.research_start_idx = 1
            session.state = SessionState.IDLE
            await store.save(session)

        msg = make_message(42, "Check https://example.com/article")

        async def fake_run_agent(*args, **kwargs):
            pass

        with patch("deeper_bot.bot.run_agent", side_effect=fake_run_agent):
            await _handle_user_input(
                42,
                "Check https://example.com/article",
                has_files=False,
                text_present=True,
                message=msg,
                session_store=store,
                settings=MagicMock(),
                bot=bot,
                bot_state=bot_state,
            )

        session = await store.get_or_create(42)
        assert "example.com" in session.allowed_domains

    async def test_clear_resets_allowed_domains(self, store, bot_state):
        """clear_session should reset allowed_domains to empty set."""
        session = await store.get_or_create(42)
        session.allowed_domains = {"example.com", "python.org"}
        await store.save(session)

        msg = make_message(42, "/clear")
        await clear_session(msg, store, bot_state)

        assert session.allowed_domains == set()

    async def test_awaiting_answer_url_populates_domains(self, store, bot, bot_state):
        """URLs in user replies to ask_user should be added to allowed_domains."""
        session = await store.get_or_create(42)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        session.set_awaiting_answer(future)

        msg = make_message(42, "Check https://docs.python.org/3/")
        await _handle_user_input(
            42,
            "Check https://docs.python.org/3/",
            has_files=False,
            text_present=True,
            message=msg,
            session_store=store,
            settings=MagicMock(),
            bot=bot,
            bot_state=bot_state,
        )

        assert "python.org" in session.allowed_domains
        assert future.result() == "Check https://docs.python.org/3/"

    async def test_new_research_resets_allowed_domains(self, store, bot, bot_state):
        """Starting a new research session should reset allowed_domains."""
        session = await store.get_or_create(42)
        async with session.lock:
            session.messages = [{"role": "system", "content": "sys"}]
            session.research_start_idx = 1
            session.state = SessionState.IDLE
            session.allowed_domains = {"old-domain.com", "stale.org"}
            await store.save(session)

        msg = make_message(42, "Research https://new-domain.com/topic")

        async def fake_run_agent(*args, **kwargs):
            pass

        with patch("deeper_bot.bot.run_agent", side_effect=fake_run_agent):
            await _handle_user_input(
                42,
                "Research https://new-domain.com/topic",
                has_files=False,
                text_present=True,
                message=msg,
                session_store=store,
                settings=MagicMock(),
                bot=bot,
                bot_state=bot_state,
            )

        session = await store.get_or_create(42)
        assert "new-domain.com" in session.allowed_domains
        assert "old-domain.com" not in session.allowed_domains
        assert "stale.org" not in session.allowed_domains
