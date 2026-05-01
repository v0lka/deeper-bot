"""Telegram-specific helpers for sending content that may exceed message limits."""

import logging

from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.types import BufferedInputFile

from deeper_bot.tools import markdown_to_telegram_html

logger = logging.getLogger(__name__)

MAX_TELEGRAM_MESSAGE_LENGTH = 4096


async def send_long_content(
    bot: Bot,
    chat_id: int,
    markdown: str,
    *,
    filename: str = "response.md",
    fallback_text: str = "Response is too long for inline display. Full text attached.",
) -> None:
    """Send markdown content, falling back to a file attachment when it exceeds Telegram limits."""
    html = markdown_to_telegram_html(markdown)
    if len(html) <= MAX_TELEGRAM_MESSAGE_LENGTH:
        await bot.send_message(chat_id, html, parse_mode=ParseMode.HTML)
    else:
        document = BufferedInputFile(markdown.encode("utf-8"), filename=filename)
        await bot.send_message(
            chat_id,
            markdown_to_telegram_html(fallback_text),
            parse_mode=ParseMode.HTML,
        )
        await bot.send_document(chat_id, document)
