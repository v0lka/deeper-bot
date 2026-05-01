"""Agent tool implementations, Markdown-to-Telegram-HTML converter, SSRF-safe HTTP client."""

from deeper_bot.tools.executor import (
    MAX_DOWNLOAD_SIZE,
    MAX_FETCH_CONTENT_LENGTH,
    MAX_TELEGRAM_INLINE_LENGTH,
    _ask_user,
    _finish,
    _generate_summary,
    _set_status,
    _summarize_web_content,
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
    "_summarize_web_content",
    "_validate_tool_args",
    "_validate_url",
    "_web_fetch",
    "_web_search",
    "close_http_client",
    "execute_tool",
    "markdown_to_telegram_html",
]
