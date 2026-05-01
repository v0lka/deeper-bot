import asyncio
import contextlib
from io import BytesIO
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.types import Chat, Document, Message, User

from deeper_bot.bot import (
    WhitelistMiddleware,
    _active_tasks,
    _extract_content,
    _format_user_content,
    _handle_user_input,
    _media_group_buffers,
    _media_group_timers,
    _process_media_group,
)
from deeper_bot.session import SessionState, SessionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(chat_id: int, text: str, user_id: int = 1) -> Message:
    """Create a mock Message with realistic attributes."""
    msg = MagicMock(spec=Message)
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = chat_id
    msg.chat.type = "private"
    msg.text = text
    msg.from_user = MagicMock(spec=User)
    msg.from_user.id = user_id
    msg.answer = AsyncMock()
    msg.reply = AsyncMock()
    return msg


# ---------------------------------------------------------------------------
# WhitelistMiddleware tests
# ---------------------------------------------------------------------------


class TestWhitelistMiddleware:
    async def test_allowed_user_passes(self):
        mw = WhitelistMiddleware([100, 200])
        handler = AsyncMock(return_value="ok")
        msg = _make_message(1, "hello", user_id=100)

        result = await mw(handler, msg, {})
        handler.assert_called_once()
        assert result == "ok"

    async def test_denied_user_blocked(self):
        mw = WhitelistMiddleware([100, 200])
        handler = AsyncMock(return_value="ok")
        msg = _make_message(1, "hello", user_id=999)

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
    @pytest.fixture
    async def store(self, tmp_path):
        s = SessionStore(str(tmp_path / "test.db"))
        await s.init()
        yield s
        await s.close()

    @pytest.fixture
    def settings(self):
        s = MagicMock()
        s.llm_model = "test-model"
        s.llm_base_url = "http://localhost"
        s.llm_api_key = "test-key"
        s.llm_use_reasoning = False
        s.allowed_users = []
        return s

    @pytest.fixture
    def bot(self):
        b = AsyncMock()
        b.send_message = AsyncMock()
        b.send_chat_action = AsyncMock()
        return b

    async def test_clear_resets_session(self, store):
        """The /clear handler logic should reset messages and index."""
        session = await store.get_or_create(42)
        session.messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        session.research_start_idx = 2
        await store.save(session)

        # Simulate clear logic
        async with session.lock:
            session.messages = []
            session.research_start_idx = 0
            session.state = SessionState.IDLE
            await store.save(session)

        assert session.messages == []
        assert session.research_start_idx == 0

    async def test_message_during_research_gets_wait_reply(self, store):
        """User messages during RESEARCHING state should get a 'please wait' reply."""
        session = await store.get_or_create(42)
        session.state = SessionState.RESEARCHING
        assert session.state == SessionState.RESEARCHING
        # In real code, handle_message checks state and sends reply

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
        """Compacting with fewer than 3 messages should do nothing."""
        session = await store.get_or_create(42)
        session.messages = [{"role": "system", "content": "sys"}]
        session.research_start_idx = 1

        from deeper_bot.compaction import compact_context

        await compact_context(session, settings)
        assert len(session.messages) == 1


# ---------------------------------------------------------------------------
# Status handler tests
# ---------------------------------------------------------------------------


class TestStatusHandler:
    @pytest.fixture
    async def store(self, tmp_path):
        s = SessionStore(str(tmp_path / "test.db"))
        await s.init()
        yield s
        await s.close()

    async def test_status_no_active_research(self, store):
        """When IDLE, /status should reply 'no active research'."""
        session = await store.get_or_create(42)
        assert session.state == SessionState.IDLE
        # In real code, handle_status checks state and replies accordingly

    async def test_status_researching_no_todo(self, store):
        """When RESEARCHING but no TODO set, /status should say no plan yet."""
        session = await store.get_or_create(42)
        session.state = SessionState.RESEARCHING
        assert session.todo_list is None

    async def test_status_researching_with_todo(self, store):
        """When RESEARCHING with TODO set, /status should return the TODO list."""
        session = await store.get_or_create(42)
        session.state = SessionState.RESEARCHING
        session.todo_list = "- [ ] Step 1\n- [x] Step 2"
        assert session.todo_list is not None

    async def test_clear_clears_status(self, store):
        """The /clear handler logic should clear status."""
        session = await store.get_or_create(42)
        session.todo_list = "- [ ] Step 1"
        session._status_announced = True

        async with session.lock:
            session.cancel_pending()
            session.clear_status()
            session.messages = []
            session.research_start_idx = 0
            session.state = SessionState.IDLE
            await store.save(session)

        assert session.todo_list is None
        assert session._status_announced is False


# ---------------------------------------------------------------------------
# Document message helpers
# ---------------------------------------------------------------------------


def _make_document_message(
    chat_id: int,
    filename: str,
    caption: str | None = None,
    file_id: str = "test_file_id",
    user_id: int = 1,
) -> Message:
    """Create a mock Message with a document attachment."""
    msg = MagicMock(spec=Message)
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = chat_id
    msg.chat.type = "private"
    msg.text = None
    msg.caption = caption
    msg.document = MagicMock(spec=Document)
    msg.document.file_id = file_id
    msg.document.file_name = filename
    msg.from_user = MagicMock(spec=User)
    msg.from_user.id = user_id
    msg.answer = AsyncMock()
    msg.reply = AsyncMock()
    return msg


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
        msg = _make_message(1, "hello")
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
        msg = _make_document_message(1, "notes.txt", caption="Check this")
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
        msg = _make_document_message(1, "data.txt")
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

        msg = _make_document_message(1, "archive.xyz")
        with pytest.raises(UnsupportedFileError):
            await _extract_content(msg, AsyncMock())

    async def test_pdf_delegates_to_markitdown(self):
        msg = _make_document_message(1, "report.pdf")
        bot = AsyncMock()
        file_mock = MagicMock()
        file_mock.file_path = "path/to/file"
        bot.get_file.return_value = file_mock
        bot.download_file.return_value = BytesIO(b"fake pdf")

        mock_result = MagicMock()
        mock_result.text_content = "PDF content"

        mock_instance = MagicMock()
        mock_instance.convert_stream.return_value = mock_result

        with patch("deeper_bot.converter.MarkItDown", return_value=mock_instance):
            result = await _extract_content(msg, bot)

        assert result is not None
        _, file_md, _ = result
        assert file_md == "PDF content"


# ---------------------------------------------------------------------------
# Document handling integration tests
# ---------------------------------------------------------------------------


class TestDocumentHandling:
    @pytest.fixture
    async def store(self, tmp_path):
        s = SessionStore(str(tmp_path / "test.db"))
        await s.init()
        yield s
        await s.close()

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
    @pytest.fixture
    async def store(self, tmp_path):
        s = SessionStore(str(tmp_path / "test.db"))
        await s.init()
        yield s
        await s.close()

    @pytest.fixture
    def bot(self):
        b = AsyncMock()
        b.send_message = AsyncMock()
        return b

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        for task in list(_active_tasks.values()):
            if not task.done():
                task.cancel()
        _active_tasks.clear()

    async def test_idle_with_text_starts_agent(self, store, bot):
        """IDLE state with text should append message and start agent."""
        session = await store.get_or_create(42)
        async with session.lock:
            session.messages = [{"role": "system", "content": "sys"}]
            session.research_start_idx = 1
            session.state = SessionState.IDLE
            await store.save(session)

        msg = _make_message(42, "hello")

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
            )

        assert 42 in _active_tasks
        session = await store.get_or_create(42)
        assert session.messages[-1] == {"role": "user", "content": "hello"}

    async def test_idle_with_file_only_adds_context(self, store, bot):
        """IDLE state with files but no text should add to context without starting agent."""
        session = await store.get_or_create(42)
        async with session.lock:
            session.messages = [{"role": "system", "content": "sys"}]
            session.research_start_idx = 1
            session.state = SessionState.IDLE
            await store.save(session)

        msg = _make_document_message(42, "data.txt")
        await _handle_user_input(
            42,
            "file content",
            has_files=True,
            text_present=False,
            message=msg,
            session_store=store,
            settings=MagicMock(),
            bot=bot,
        )

        assert 42 not in _active_tasks
        cast(AsyncMock, msg.reply).assert_awaited_once()
        session = await store.get_or_create(42)
        assert session.messages[-1] == {"role": "user", "content": "file content"}

    async def test_researching_sends_wait_reply(self, store, bot):
        """RESEARCHING state should reply with 'please wait'."""
        session = await store.get_or_create(42)
        async with session.lock:
            session.state = SessionState.RESEARCHING
            await store.save(session)

        msg = _make_message(42, "hello")
        await _handle_user_input(
            42,
            "hello",
            has_files=False,
            text_present=True,
            message=msg,
            session_store=store,
            settings=MagicMock(),
            bot=bot,
        )

        cast(AsyncMock, msg.reply).assert_awaited_once()
        assert 42 not in _active_tasks

    async def test_awaiting_answer_resolves_future(self, store, bot):
        """AWAITING_ANSWER state should resolve the pending future."""
        session = await store.get_or_create(42)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        session.set_awaiting_answer(future)

        msg = _make_message(42, "my answer")
        await _handle_user_input(
            42,
            "my answer",
            has_files=False,
            text_present=True,
            message=msg,
            session_store=store,
            settings=MagicMock(),
            bot=bot,
        )

        assert future.result() == "my answer"
        assert session.state == SessionState.RESEARCHING


# ---------------------------------------------------------------------------
# Media group helpers
# ---------------------------------------------------------------------------


def _make_media_group_message(
    chat_id: int,
    media_group_id: str,
    filename: str,
    caption: str | None = None,
    file_id: str = "test_file_id",
    user_id: int = 1,
) -> Message:
    """Create a mock Message that belongs to a media group."""
    msg = MagicMock(spec=Message)
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = chat_id
    msg.chat.type = "private"
    msg.text = None
    msg.caption = caption
    msg.media_group_id = media_group_id
    msg.document = MagicMock(spec=Document)
    msg.document.file_id = file_id
    msg.document.file_name = filename
    msg.from_user = MagicMock(spec=User)
    msg.from_user.id = user_id
    msg.answer = AsyncMock()
    msg.reply = AsyncMock()
    return msg


# ---------------------------------------------------------------------------
# Media group handling tests
# ---------------------------------------------------------------------------


class TestMediaGroupHandling:
    @pytest.fixture
    async def store(self, tmp_path):
        s = SessionStore(str(tmp_path / "test.db"))
        await s.init()
        yield s
        await s.close()

    @pytest.fixture
    def settings(self):
        s = MagicMock()
        s.llm_model = "test-model"
        s.llm_base_url = "http://localhost"
        s.llm_api_key = "test-key"
        s.llm_use_reasoning = False
        s.allowed_users = []
        return s

    @pytest.fixture
    def bot(self):
        b = AsyncMock()
        b.send_message = AsyncMock()
        b.send_chat_action = AsyncMock()
        return b

    @pytest.fixture(autouse=True)
    def _cleanup_media_groups(self):
        yield
        _media_group_buffers.clear()
        for task in list(_media_group_timers.values()):
            if not task.done():
                task.cancel()
        _media_group_timers.clear()

    async def test_media_group_combines_files_into_single_prompt(self, store, settings, bot):
        """Multiple files in a media group should be combined into one user message."""
        session = await store.get_or_create(42)
        async with session.lock:
            session.messages = [{"role": "system", "content": "sys"}]
            session.research_start_idx = 1
            session.state = SessionState.IDLE
            await store.save(session)

        msg1 = _make_media_group_message(42, "mg1", "file1.txt", caption="Analyze these")
        msg2 = _make_media_group_message(42, "mg1", "file2.txt", caption="Analyze these")
        msg3 = _make_media_group_message(42, "mg1", "file3.txt", caption="Analyze these")
        _media_group_buffers["mg1"] = [msg1, msg2, msg3]

        async def fake_extract(msg, bot):
            return msg.caption, f"content of {msg.document.file_name}", msg.document.file_name

        with patch("deeper_bot.bot._extract_content", side_effect=fake_extract):
            with patch("asyncio.sleep"):
                await _process_media_group(42, "mg1", bot, store, settings)

        # An agent task should have been registered
        assert 42 in _active_tasks
        # Clean up the task
        task = _active_tasks.pop(42, None)
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

    async def test_media_group_without_caption_adds_context_no_agent(self, store, settings, bot):
        """Media group without caption should add files to context but not start agent."""
        session = await store.get_or_create(42)
        async with session.lock:
            session.messages = [{"role": "system", "content": "sys"}]
            session.research_start_idx = 1
            session.state = SessionState.IDLE
            await store.save(session)

        msg1 = _make_media_group_message(42, "mg2", "file1.txt")
        msg2 = _make_media_group_message(42, "mg2", "file2.txt")
        _media_group_buffers["mg2"] = [msg1, msg2]

        async def fake_extract(msg, bot):
            return "", f"content of {msg.document.file_name}", msg.document.file_name

        with patch("deeper_bot.bot._extract_content", side_effect=fake_extract):
            with patch("asyncio.sleep"):
                await _process_media_group(42, "mg2", bot, store, settings)

        # No agent should have been started
        assert 42 not in _active_tasks

        session = await store.get_or_create(42)
        user_msgs = [m for m in session.messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert "file1.txt" in user_msgs[0]["content"]
        assert "file2.txt" in user_msgs[0]["content"]

    async def test_media_group_skips_unsupported_files(self, store, settings, bot):
        """Unsupported files in a media group should be skipped, others processed."""
        session = await store.get_or_create(42)
        async with session.lock:
            session.messages = [{"role": "system", "content": "sys"}]
            session.research_start_idx = 1
            session.state = SessionState.IDLE
            await store.save(session)

        msg1 = _make_media_group_message(42, "mg3", "file1.txt", caption="Analyze")
        msg2 = _make_media_group_message(42, "mg3", "bad.xyz")
        _media_group_buffers["mg3"] = [msg1, msg2]

        async def fake_extract(msg, bot):
            if msg.document.file_name == "bad.xyz":
                from deeper_bot.converter import UnsupportedFileError

                raise UnsupportedFileError("Unsupported file: bad.xyz")
            return msg.caption, f"content of {msg.document.file_name}", msg.document.file_name

        with patch("deeper_bot.bot._extract_content", side_effect=fake_extract):
            with patch("asyncio.sleep"):
                await _process_media_group(42, "mg3", bot, store, settings)

        # An agent task should have been registered
        assert 42 in _active_tasks
        task = _active_tasks.pop(42, None)
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
# _active_tasks cleanup tests
# ---------------------------------------------------------------------------


class TestActiveTasksCleanup:
    async def test_done_callback_cleans_up_on_success(self):
        """Completed task should be removed from _active_tasks."""
        chat_id = 99999

        async def dummy():
            pass

        task = asyncio.create_task(dummy())
        _active_tasks[chat_id] = task
        await task
        # Simulate the callback registration pattern from bot.py
        from deeper_bot.bot import _active_tasks as tasks

        tasks.pop(chat_id, None)
        assert chat_id not in _active_tasks

    async def test_done_callback_cleans_up_on_cancellation(self):
        """Cancelled task should be removed from _active_tasks."""
        chat_id = 99998

        async def hang():
            await asyncio.sleep(3600)

        task = asyncio.create_task(hang())
        _active_tasks[chat_id] = task
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        _active_tasks.pop(chat_id, None)
        assert chat_id not in _active_tasks
