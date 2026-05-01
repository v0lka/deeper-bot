"""Shared test helper functions for deeper-bot tests."""

import json
from unittest.mock import AsyncMock, MagicMock

from aiogram.types import Chat, Document, Message, User


def make_message(chat_id: int, text: str, user_id: int = 1, language_code: str | None = None) -> Message:
    """Create a mock Message with realistic attributes."""
    msg = MagicMock(spec=Message)
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = chat_id
    msg.chat.type = "private"
    msg.text = text
    msg.document = None
    msg.from_user = MagicMock(spec=User)
    msg.from_user.id = user_id
    msg.from_user.language_code = language_code
    msg.answer = AsyncMock()
    msg.reply = AsyncMock()
    msg.media_group_id = None
    return msg


def make_document_message(
    chat_id: int,
    filename: str,
    caption: str | None = None,
    file_id: str = "test_file_id",
    user_id: int = 1,
    language_code: str | None = None,
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
    msg.from_user.language_code = language_code
    msg.answer = AsyncMock()
    msg.reply = AsyncMock()
    msg.media_group_id = None
    return msg


def make_media_group_message(
    chat_id: int,
    media_group_id: str,
    filename: str,
    caption: str | None = None,
    file_id: str = "test_file_id",
    user_id: int = 1,
    language_code: str | None = None,
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
    msg.from_user.language_code = language_code
    msg.answer = AsyncMock()
    msg.reply = AsyncMock()
    return msg


def make_tool_call(name: str, arguments: dict, call_id: str = "call_1"):
    """Create a mock tool call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc
