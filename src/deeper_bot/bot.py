import asyncio
import logging
from typing import Any

from aiogram import BaseMiddleware, Bot, Router
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import BotCommand, Message, TelegramObject

from deeper_bot.agent import run_agent
from deeper_bot.compaction import compact_context
from deeper_bot.config import Settings
from deeper_bot.converter import ConversionError, UnsupportedFileError, convert_file, is_supported
from deeper_bot.prompts import SYSTEM_PROMPT
from deeper_bot.session import SessionState, SessionStore
from deeper_bot.tools import markdown_to_telegram_html

logger = logging.getLogger(__name__)

_active_tasks: dict[int, asyncio.Task] = {}

# Media group buffering (Telegram sends each file in a group as a separate message)
_media_group_buffers: dict[str, list[Message]] = {}
_media_group_timers: dict[str, asyncio.Task] = {}
_MEDIA_GROUP_DELAY = 1.5

FILES_ADDED_MSG = "File added to context. Send your research question or instructions."

UNSUPPORTED_FORMAT_MSG = (
    "Unsupported file format. Supported formats: PDF, DOCX, XLSX, PPTX, TXT, MD, and common code files."
)

CONVERSION_FAILED_MSG = "Failed to process the attached file. Please check the file is not corrupted."

WAIT_INIT_MSG = "Wait for initialization please (it may take a few seconds) 🙏"


# ---------------------------------------------------------------------------
# Content extraction helpers
# ---------------------------------------------------------------------------


async def _extract_content(message: Message, bot: Bot) -> tuple[str, str | None, str | None] | None:
    """Extract text and optional file content from a message.

    Returns (text, file_markdown, filename) or None if message should be ignored.
    """
    if message.document:
        filename = message.document.file_name or "file"
        if not is_supported(filename):
            raise UnsupportedFileError(f"Unsupported file: {filename}")
        file = await bot.get_file(message.document.file_id)
        if file.file_path is None:
            raise ConversionError("File path unavailable from Telegram.")
        data = await bot.download_file(file.file_path)
        if data is None:
            raise ConversionError("Failed to download file from Telegram.")
        file_markdown = await convert_file(data, filename)
        text = message.caption or ""
        return text, file_markdown, filename

    if message.text:
        return message.text, None, None

    return None


def _format_user_content(text: str, file_markdown: str, filename: str) -> str:
    """Format user message content with attached file."""
    header = f"Attached file: `{filename}`"
    if text.strip():
        return f"{text}\n\n---\n\n{header}\n\n{file_markdown}"
    return f"{header}\n\n{file_markdown}"


async def _handle_user_input(
    chat_id: int,
    user_content: str,
    has_files: bool,
    text_present: bool,
    message: Message,
    session_store: SessionStore,
    settings: Settings,
    bot: Bot,
) -> None:
    """Add user content to session and start agent if appropriate."""
    session = await session_store.get_or_create(chat_id)

    async with session.lock:
        if session.state == SessionState.AWAITING_ANSWER:
            session.resolve_answer(user_content)
            return

        if session.state == SessionState.RESEARCHING:
            await message.reply(
                markdown_to_telegram_html("Research in progress, please wait. Use /status to check progress."),
                parse_mode=ParseMode.HTML,
            )
            return

        # IDLE — start new research
        # Ensure system prompt is present
        if not session.messages:
            session.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            session.research_start_idx = 1

        session.messages.append({"role": "user", "content": user_content})

        if session.research_start_idx >= len(session.messages):
            session.research_start_idx = len(session.messages) - 1

        # File(s) without text — add to context but don't start agent
        if has_files and not text_present:
            await session_store.save(session)
            await message.reply(markdown_to_telegram_html(FILES_ADDED_MSG), parse_mode=ParseMode.HTML)
            return

        session.state = SessionState.RESEARCHING
        await session_store.save(session)

        if not session._initialized:
            await message.reply(
                markdown_to_telegram_html(WAIT_INIT_MSG),
                parse_mode=ParseMode.HTML,
            )

        task = asyncio.create_task(run_agent(session, bot, chat_id, settings, session_store))
        _active_tasks[chat_id] = task

        def _on_task_done(t: asyncio.Task) -> None:
            try:
                _active_tasks.pop(chat_id, None)
                if t.cancelled():
                    return
                exc = t.exception()
                if exc:
                    logger.error("Agent task for chat_id=%d failed: %s", chat_id, exc)
            except Exception:
                logger.exception("Error in task done callback for chat_id=%d", chat_id)

        task.add_done_callback(_on_task_done)


async def _process_media_group(
    chat_id: int,
    media_group_id: str,
    bot: Bot,
    session_store: SessionStore,
    settings: Settings,
) -> None:
    """Wait for all messages in a media group, then process them as a single prompt."""
    await asyncio.sleep(_MEDIA_GROUP_DELAY)
    messages = _media_group_buffers.pop(media_group_id, [])
    _media_group_timers.pop(media_group_id, None)

    if not messages:
        return

    file_parts: list[tuple[str, str]] = []
    caption: str | None = None
    errors: list[str] = []

    for msg in messages:
        if not msg.document:
            continue
        try:
            result = await _extract_content(msg, bot)
        except UnsupportedFileError as e:
            errors.append(str(e))
            continue
        except ConversionError as e:
            errors.append(str(e))
            continue
        except Exception:
            logger.exception("Failed to process message in media group %s", media_group_id)
            errors.append("Failed to process a file.")
            continue

        if result is None:
            continue

        text, file_markdown, filename = result
        if text and caption is None:
            caption = text
        if file_markdown and filename:
            file_parts.append((filename, file_markdown))

    # Report any extraction errors
    if errors:
        error_text = "\n".join(f"- {e}" for e in errors)
        try:
            await messages[0].reply(
                markdown_to_telegram_html(f"Some files could not be processed:\n{error_text}"),
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            logger.exception("Failed to send media group error reply")

    if not file_parts and not caption:
        return

    # Build combined user content from all files
    parts: list[str] = []
    if caption:
        parts.append(caption)
    for filename, file_markdown in file_parts:
        parts.append(f"\n\n---\n\nAttached file: `{filename}`\n\n{file_markdown}")
    user_content = "".join(parts)

    await _handle_user_input(
        chat_id,
        user_content,
        has_files=True,
        text_present=bool(caption and caption.strip()),
        message=messages[0],
        session_store=session_store,
        settings=settings,
        bot=bot,
    )


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class WhitelistMiddleware(BaseMiddleware):
    def __init__(self, allowed_users: list[int]) -> None:
        self._allowed = set(allowed_users)

    async def __call__(
        self,
        handler: Any,
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if isinstance(event, Message) and event.from_user and event.from_user.id not in self._allowed:
            logger.debug("Ignoring message from non-whitelisted user %s", event.from_user.id)
            return None
        return await handler(event, data)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def create_router() -> Router:
    router = Router(name="deeper_bot")

    @router.message(Command("clear"))
    async def handle_clear(message: Message, session_store: SessionStore, **kwargs: Any) -> None:
        chat_id = message.chat.id

        session = await session_store.get_or_create(chat_id)
        async with session.lock:
            task = _active_tasks.pop(chat_id, None)
            if task and not task.done():
                task.cancel()
            session.cancel_pending()
            session.clear_status()

            session.messages = []
            session.research_start_idx = 0
            session.state = SessionState.IDLE
            await session_store.save(session)

        await message.answer(markdown_to_telegram_html("Context cleared."), parse_mode=ParseMode.HTML)

    @router.message(Command("compact"))
    async def handle_compact(message: Message, session_store: SessionStore, settings: Settings, **kwargs: Any) -> None:
        chat_id = message.chat.id

        session = await session_store.get_or_create(chat_id)
        async with session.lock:
            if session.research_start_idx <= 1:
                await message.answer(markdown_to_telegram_html("Nothing to compact."), parse_mode=ParseMode.HTML)
                return

            await compact_context(session, settings)
            await session_store.save(session)

        await message.answer(markdown_to_telegram_html("Context compacted."), parse_mode=ParseMode.HTML)

    @router.message(Command("status"))
    async def handle_status(message: Message, session_store: SessionStore, **kwargs: Any) -> None:
        chat_id = message.chat.id
        session = await session_store.get_or_create(chat_id)
        if session.state not in (SessionState.RESEARCHING, SessionState.AWAITING_ANSWER):
            await message.answer(markdown_to_telegram_html("No active research session."), parse_mode=ParseMode.HTML)
            return
        if session.todo_list is None:
            await message.answer(
                markdown_to_telegram_html("Research is in progress, but no plan has been set yet."),
                parse_mode=ParseMode.HTML,
            )
            return
        html = markdown_to_telegram_html(f"TODO:\n\n{session.todo_list}")
        await message.answer(html, parse_mode=ParseMode.HTML)

    @router.message()
    async def handle_message(message: Message, session_store: SessionStore, settings: Settings, bot: Bot) -> None:
        if message.chat.type != "private":
            return
        if not message.text and not message.document:
            return

        chat_id = message.chat.id

        # Buffer media groups so multiple files sent together are combined into one prompt
        if message.media_group_id:
            mg_id = message.media_group_id
            if mg_id not in _media_group_buffers:
                _media_group_buffers[mg_id] = []
                _media_group_timers[mg_id] = asyncio.create_task(
                    _process_media_group(chat_id, mg_id, bot, session_store, settings)
                )
            _media_group_buffers[mg_id].append(message)
            return

        # Single message handling
        try:
            result = await _extract_content(message, bot)
        except UnsupportedFileError:
            await message.reply(markdown_to_telegram_html(UNSUPPORTED_FORMAT_MSG), parse_mode=ParseMode.HTML)
            return
        except ConversionError:
            await message.reply(markdown_to_telegram_html(CONVERSION_FAILED_MSG), parse_mode=ParseMode.HTML)
            return
        except Exception:
            logger.exception("Failed to process message content for chat_id=%d", chat_id)
            await message.reply(
                markdown_to_telegram_html("Failed to download or process the file. Please try again."),
                parse_mode=ParseMode.HTML,
            )
            return

        if result is None:
            return

        text, file_markdown, filename = result
        user_content = _format_user_content(text, file_markdown, filename) if file_markdown and filename else text
        has_files = file_markdown is not None and filename is not None
        text_present = bool(text and text.strip())

        await _handle_user_input(
            chat_id,
            user_content,
            has_files,
            text_present,
            message,
            session_store,
            settings,
            bot,
        )

    return router


async def on_startup(bot: Bot) -> None:
    await bot.set_my_commands(
        [
            BotCommand(command="clear", description="Clear context"),
            BotCommand(command="compact", description="Compact context"),
            BotCommand(command="status", description="Show current research progress"),
        ]
    )


def setup_router(router: Router, settings: Settings) -> None:
    """Apply middleware and startup hooks to the router."""
    if settings.allowed_users:
        router.message.middleware(WhitelistMiddleware(settings.allowed_users))
