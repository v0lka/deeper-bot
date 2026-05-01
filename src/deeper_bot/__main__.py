"""Application entry point for Deeper Bot."""

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from deeper_bot.bot import BotState, create_router, on_startup, setup_router
from deeper_bot.config import get_settings
from deeper_bot.session import SessionStore
from deeper_bot.tools import close_http_client


async def main() -> None:
    """Initialize and start the Telegram bot."""
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = get_settings()

    store = SessionStore(settings.database_path)
    await store.init()

    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        base_url=settings.telegram_api_url,
    )
    dp = Dispatcher()

    dp["session_store"] = store
    dp["settings"] = settings
    dp["bot_state"] = BotState()

    router = create_router()
    setup_router(router, settings)
    dp.include_router(router)

    dp.startup.register(on_startup)

    try:
        await dp.start_polling(bot)
    finally:
        await close_http_client()
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
