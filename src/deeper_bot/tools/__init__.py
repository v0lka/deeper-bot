"""Agent tool implementations, Markdown-to-Telegram-HTML converter, SSRF-safe HTTP client."""

from deeper_bot.tools.documents import (
    MAX_FETCH_CONTENT_LENGTH,
    clear_session_documents,
    format_document_response,
    read_document_fragment,
)
from deeper_bot.tools.executor import (
    MAX_DOWNLOAD_SIZE,
    MAX_TELEGRAM_INLINE_LENGTH,
    _ask_user,
    _finish,
    _generate_summary,
    _set_status,
    _web_fetch,
    _web_search,
    execute_tool,
)
from deeper_bot.tools.http import (
    SSRFBlockedError,
    _get_http_client,
    _is_ip_allowed,
    _SSRFSafeTransport,
    _validate_url,
    close_http_client,
)
from deeper_bot.tools.markdown import TelegramHTMLRenderer, markdown_to_telegram_html
from deeper_bot.tools.schemas import (
    TOOLS,
    AskUserArgs,
    FinishArgs,
    ReadDocumentArgs,
    SetStatusArgs,
    WebFetchArgs,
    WebSearchArgs,
    _validate_tool_args,
)

__all__ = [
    "TOOLS",
    "AskUserArgs",
    "FinishArgs",
    "MAX_DOWNLOAD_SIZE",
    "MAX_FETCH_CONTENT_LENGTH",
    "MAX_TELEGRAM_INLINE_LENGTH",
    "ReadDocumentArgs",
    "SSRFBlockedError",
    "SetStatusArgs",
    "TelegramHTMLRenderer",
    "WebFetchArgs",
    "WebSearchArgs",
    "_SSRFSafeTransport",
    "_ask_user",
    "_finish",
    "_generate_summary",
    "_get_http_client",
    "_is_ip_allowed",
    "_set_status",
    "_validate_tool_args",
    "_validate_url",
    "_web_fetch",
    "_web_search",
    "clear_session_documents",
    "close_http_client",
    "execute_tool",
    "format_document_response",
    "markdown_to_telegram_html",
    "read_document_fragment",
]
