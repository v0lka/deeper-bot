import asyncio
import contextlib
import logging

import litellm
from aiogram import Bot
from aiogram.enums import ChatAction, ParseMode
from aiogram.types import BufferedInputFile

from deeper_bot.compaction import compact_context
from deeper_bot.config import Settings
from deeper_bot.llm import llm_call_with_retry
from deeper_bot.session import Session, SessionState, SessionStore
from deeper_bot.tools import TOOLS, execute_tool, markdown_to_telegram_html

logger = logging.getLogger(__name__)

_MAX_TELEGRAM_MESSAGE_LENGTH = 4096
MAX_AGENT_ITERATIONS = 50
MAX_COMPACTION_RETRIES = 2


async def _keep_typing(bot: Bot, chat_id: int) -> None:
    """Send typing indicator every 4 seconds until cancelled."""
    try:
        while True:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


async def run_agent(
    session: Session,
    bot: Bot,
    chat_id: int,
    settings: Settings,
    session_store: SessionStore,
) -> None:
    """Run the ReAct agent loop for a research session."""
    try:
        await _agent_loop(session, bot, chat_id, settings, session_store)
    except asyncio.CancelledError:
        logger.info("Agent loop cancelled for chat_id=%d", chat_id)
    except Exception:
        logger.exception("Unhandled error in agent loop for chat_id=%d", chat_id)
        with contextlib.suppress(Exception):
            await bot.send_message(
                chat_id,
                markdown_to_telegram_html("An unexpected error occurred. The research session has been stopped."),
                parse_mode=ParseMode.HTML,
            )
    finally:
        session.clear_status()
        if session.state != SessionState.IDLE:
            session.state = SessionState.IDLE
        await session_store.save(session)


async def _agent_loop(
    session: Session,
    bot: Bot,
    chat_id: int,
    settings: Settings,
    session_store: SessionStore,
) -> None:
    compaction_retries = 0
    for _iteration in range(MAX_AGENT_ITERATIONS):
        typing_task = asyncio.create_task(_keep_typing(bot, chat_id))
        try:
            messages_for_llm = list(session.messages)
            if session.todo_list is not None:
                messages_for_llm.append(
                    {
                        "role": "system",
                        "content": f"## Current Research Progress\n\n{session.todo_list}",
                    }
                )

            kwargs: dict = {
                "model": settings.llm_model,
                "messages": messages_for_llm,
                "tools": TOOLS,
                "api_base": settings.llm_base_url,
                "api_key": settings.resolved_llm_api_key,
            }
            if settings.llm_use_reasoning:
                kwargs["reasoning_effort"] = settings.llm_reasoning_effort

            response = await llm_call_with_retry(kwargs)
        except litellm.ContextWindowExceededError:
            compaction_retries += 1
            if compaction_retries > MAX_COMPACTION_RETRIES:
                logger.warning(
                    "Context too large after %d compaction attempts for chat_id=%d",
                    compaction_retries,
                    chat_id,
                )
                await bot.send_message(
                    chat_id,
                    markdown_to_telegram_html(
                        "Context is too large even after compaction. Use /clear to start a new session."
                    ),
                    parse_mode=ParseMode.HTML,
                )
                return
            logger.info("Context window exceeded for chat_id=%d, compacting (attempt %d)", chat_id, compaction_retries)
            await bot.send_message(
                chat_id, markdown_to_telegram_html("Context window exceeded. Compacting..."), parse_mode=ParseMode.HTML
            )
            await compact_context(session, settings)
            await session_store.save(session)
            continue
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("LLM call error for chat_id=%d", chat_id)
            await bot.send_message(
                chat_id,
                markdown_to_telegram_html("A temporary error occurred. Please try again."),
                parse_mode=ParseMode.HTML,
            )
            return
        finally:
            typing_task.cancel()

        msg = response.choices[0].message
        session.messages.append(msg.model_dump(exclude_none=True))
        session._initialized = True

        if msg.tool_calls:
            finish_called = False
            for tc in msg.tool_calls:
                result, is_finish = await execute_tool(tc, session, bot, chat_id, settings)
                session.messages.append(result)
                if is_finish:
                    finish_called = True

            await session_store.save(session)

            if finish_called:
                session.state = SessionState.IDLE
                session.research_start_idx = len(session.messages)
                await session_store.save(session)
                return
        else:
            # No tool calls — model responded with text directly
            if msg.content:
                try:
                    html_content = markdown_to_telegram_html(msg.content)
                    if len(html_content) <= _MAX_TELEGRAM_MESSAGE_LENGTH:
                        await bot.send_message(chat_id, html_content, parse_mode=ParseMode.HTML)
                    else:
                        file_bytes = msg.content.encode("utf-8")
                        document = BufferedInputFile(file_bytes, filename="response.md")
                        await bot.send_message(
                            chat_id,
                            markdown_to_telegram_html("Response is too long for inline display. Full text attached."),
                            parse_mode=ParseMode.HTML,
                        )
                        await bot.send_document(chat_id, document)
                except Exception:
                    logger.exception("Failed to send assistant message")
            session.state = SessionState.IDLE
            session.research_start_idx = len(session.messages)
            await session_store.save(session)
            return

    # Exhausted MAX_AGENT_ITERATIONS
    logger.warning("Agent loop exceeded %d iterations for chat_id=%d", MAX_AGENT_ITERATIONS, chat_id)
    await bot.send_message(
        chat_id,
        markdown_to_telegram_html(
            "Research stopped: exceeded maximum number of iterations. Use /status to see progress."
        ),
        parse_mode=ParseMode.HTML,
    )
