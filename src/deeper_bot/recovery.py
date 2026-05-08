"""Session recovery: detect and resume sessions interrupted by a crash or restart."""

import asyncio
import logging

from aiogram import Bot

from deeper_bot.agent import run_agent
from deeper_bot.bot import BotState, make_task_done_callback
from deeper_bot.config import Settings
from deeper_bot.session import SessionState, SessionStore

logger = logging.getLogger(__name__)

RECOVERY_STAGGER_DELAY = 5.0

RECOVERY_TOOL_CONTENT = (
    "Tool execution was interrupted by a system restart. Continue your research from where you left off."
)


def repair_message_history(messages: list[dict]) -> None:
    """Fix broken message history where trailing tool_calls lack corresponding results.

    Mutates *messages* in place by appending synthetic tool result messages for any
    tool_call_ids that don't have a matching role=tool entry after the last assistant
    message with tool_calls.
    """
    if not messages:
        return

    # Find the last assistant message with tool_calls (walk backwards)
    assistant_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            assistant_idx = i
            break

    if assistant_idx is None:
        return

    # Idempotency guard: check if recovery content already present in trailing tool results
    for j in range(assistant_idx + 1, len(messages)):
        if messages[j].get("role") == "tool" and messages[j].get("content") == RECOVERY_TOOL_CONTENT:
            return

    # Collect expected tool_call_ids
    tool_calls = messages[assistant_idx]["tool_calls"]
    expected_ids = [tc["id"] for tc in tool_calls]

    # Collect actual tool results that follow the assistant message
    actual_ids: set[str] = set()
    for j in range(assistant_idx + 1, len(messages)):
        if messages[j].get("role") == "tool":
            actual_ids.add(messages[j].get("tool_call_id", ""))
        else:
            break

    # Append synthetic results for missing tool_call_ids
    for tc_id in expected_ids:
        if tc_id not in actual_ids:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": RECOVERY_TOOL_CONTENT,
                }
            )


async def recover_sessions(
    session_store: SessionStore,
    bot: Bot,
    settings: Settings,
    bot_state: BotState,
) -> None:
    """Detect and resume sessions that were interrupted by a crash or restart.

    Launches agent loops sequentially with a stagger delay between each.
    """
    try:
        chat_ids = await session_store.get_interrupted_chat_ids()
    except Exception:
        logger.exception("Failed to query interrupted sessions")
        return

    if not chat_ids:
        logger.info("No interrupted sessions to recover")
        return

    logger.info("Recovering %d interrupted session(s)", len(chat_ids))

    recovered = 0
    for chat_id in chat_ids:
        try:
            session = await session_store.get_or_create(chat_id)

            async with session.lock:
                # Skip if already transitioned to IDLE (e.g. by concurrent /clear)
                if session.state == SessionState.IDLE:
                    continue

                # Guard: nothing to resume if messages are empty/minimal
                if len(session.messages) <= 1:
                    session.state = SessionState.IDLE
                    await session_store.save(session)
                    continue

                # Repair broken message history
                repair_message_history(session.messages)

                # AWAITING_ANSWER → RESEARCHING (Future is unrecoverable)
                if session.state == SessionState.AWAITING_ANSWER:
                    session._pending_future = None
                    session.state = SessionState.RESEARCHING

                await session_store.save(session)

            # Launch agent loop
            task = asyncio.create_task(run_agent(session, bot, chat_id, settings, session_store))
            bot_state.active_tasks[chat_id] = task
            task.add_done_callback(make_task_done_callback(chat_id, bot_state))
            recovered += 1

        except Exception:
            logger.exception("Failed to recover session for chat_id=%d", chat_id)
            continue

        await asyncio.sleep(RECOVERY_STAGGER_DELAY)

    logger.info("Session recovery complete: %d session(s) resumed", recovered)
