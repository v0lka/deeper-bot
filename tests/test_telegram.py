"""Tests for deeper_bot.telegram module."""

from unittest.mock import AsyncMock, patch

from deeper_bot.telegram import MAX_TELEGRAM_MESSAGE_LENGTH, send_long_content


class TestSendLongContent:
    async def test_short_content_sent_inline(self):
        bot = AsyncMock()
        await send_long_content(bot, 42, "Hello world")
        bot.send_message.assert_awaited_once()
        assert bot.send_document.await_count == 0

    async def test_long_content_sent_as_file(self):
        bot = AsyncMock()
        long_md = "x" * (MAX_TELEGRAM_MESSAGE_LENGTH + 100)
        with patch("deeper_bot.telegram.markdown_to_telegram_html", side_effect=lambda t: t):
            await send_long_content(bot, 42, long_md)
        assert bot.send_message.await_count == 1
        bot.send_document.assert_awaited_once()

    async def test_custom_filename(self):
        bot = AsyncMock()
        long_md = "x" * (MAX_TELEGRAM_MESSAGE_LENGTH + 100)
        with patch("deeper_bot.telegram.markdown_to_telegram_html", side_effect=lambda t: t):
            await send_long_content(bot, 42, long_md, filename="report.md")
        doc_call = bot.send_document.await_args
        assert doc_call[0][1].filename == "report.md"

    async def test_custom_fallback_text(self):
        bot = AsyncMock()
        long_md = "x" * (MAX_TELEGRAM_MESSAGE_LENGTH + 100)
        with patch("deeper_bot.telegram.markdown_to_telegram_html", side_effect=lambda t: t):
            await send_long_content(bot, 42, long_md, fallback_text="Too long!")
        msg_call = bot.send_message.await_args
        assert msg_call[0][1] == "Too long!"
