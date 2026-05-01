"""Agent tool implementations, Markdown-to-Telegram-HTML converter, SSRF-safe HTTP client."""

import asyncio
import contextlib
import ipaddress
import json
import logging
import socket
import urllib.parse
from html import escape as html_escape
from typing import Any, cast

import httpx
import mistune
import trafilatura
from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramForbiddenError
from aiogram.types import BufferedInputFile
from ddgs import DDGS
from litellm.types.utils import ChatCompletionMessageToolCall
from mistune.core import BlockState
from pydantic import BaseModel, Field, ValidationError

from deeper_bot.config import Settings
from deeper_bot.llm import llm_call_with_retry
from deeper_bot.session import Session

logger = logging.getLogger(__name__)

MAX_FETCH_CONTENT_LENGTH = 15_000
MAX_TELEGRAM_INLINE_LENGTH = 2000
MAX_DOWNLOAD_SIZE = 5_000_000

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo. Returns a list of results with title, URL, and snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": 5,
                        "maximum": 15,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch a web page and extract its main content as Markdown."
                " Use for reading full articles from search results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "Ask the user a clarifying question and wait for their response."
                " The user has up to 60 minutes to respond before the request times out."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to ask the user."},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_status",
            "description": (
                "Set or update the current research progress TODO list."
                " Use Markdown checkboxes: '- [ ]' for pending, '- [X]' for done."
                " Call as your FIRST action to announce the plan, then update as you complete steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todo_list": {
                        "type": "string",
                        "description": "The research TODO list in Markdown with checkboxes.",
                    },
                },
                "required": ["todo_list"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Finalize the research and deliver the complete report."
                " Provide the full research result in Markdown format."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "result_markdown": {
                        "type": "string",
                        "description": "The complete research report in Markdown format.",
                    },
                },
                "required": ["result_markdown"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Markdown -> Telegram HTML converter
# ---------------------------------------------------------------------------


class TelegramHTMLRenderer(mistune.HTMLRenderer):
    """Renders Markdown to Telegram-compatible HTML subset.

    Telegram supports: b, strong, i, em, u, ins, s, strike, del,
    code, pre, a, blockquote, tg-spoiler.
    """

    NAME = "telegram_html"

    def __init__(self) -> None:
        """Initialize the renderer with list tracking state."""
        super().__init__()
        self._ordered: bool | None = None
        self._item_index: int = 1

    def render_token(self, token: dict[str, Any], state: BlockState) -> str:
        """Render a single token, tracking list ordering state."""
        if token["type"] == "list":
            prev_ordered = self._ordered
            prev_index = self._item_index
            attrs = token.get("attrs", {})
            self._ordered = attrs.get("ordered", False)
            self._item_index = attrs.get("start", 1)
            result = super().render_token(token, state)
            self._ordered = prev_ordered
            self._item_index = prev_index
            return result
        return super().render_token(token, state)

    def paragraph(self, text: str) -> str:
        """Render a paragraph."""
        return f"{text}\n\n"

    def strong(self, text: str) -> str:
        """Render bold text."""
        return f"<b>{text}</b>"

    def emphasis(self, text: str) -> str:
        """Render italic text."""
        return f"<i>{text}</i>"

    def heading(self, text: str, level: int, **attrs: Any) -> str:
        """Render a heading as bold text."""
        return f"<b>{text}</b>\n\n"

    def block_code(self, code: str, info: str | None = None) -> str:
        """Render a code block with optional language info."""
        if info:
            return f'<pre><code class="language-{html_escape(info)}">{html_escape(code)}</code></pre>\n'
        return f"<pre>{html_escape(code)}</pre>\n"

    def block_quote(self, text: str) -> str:
        """Render a blockquote."""
        return f"<blockquote>{text}</blockquote>\n"

    def list(self, text: str, ordered: bool, **attrs: Any) -> str:
        """Render a list wrapper."""
        return text + "\n"

    def list_item(self, text: str) -> str:
        """Render a list item with ordered or bullet prefix."""
        if self._ordered:
            prefix = f"{self._item_index}. "
            self._item_index += 1
        else:
            prefix = "\u2022 "
        return f"{prefix}{text.strip()}\n"

    def image(self, text: str, url: str, title: str | None = None) -> str:
        """Render an image as a link."""
        return f'<a href="{self.safe_url(url)}">{text or "image"}</a>'

    def linebreak(self) -> str:
        """Render a hard line break."""
        return "\n"

    def softbreak(self) -> str:
        """Render a soft line break."""
        return "\n"

    def thematic_break(self) -> str:
        """Render a thematic break (horizontal rule)."""
        return "\n"

    def table(self, text: str) -> str:
        """Render a table wrapper."""
        return text.strip() + "\n\n"

    def table_head(self, text: str) -> str:
        """Render a table header as bold lines."""
        lines = []
        for line in text.splitlines():
            line = line.rstrip(" |")
            if line:
                lines.append(line)
        if lines:
            return "\n".join(lines) + "\n"
        return ""

    def table_body(self, text: str) -> str:
        """Render a table body as bulleted lines."""
        lines = []
        for line in text.splitlines():
            line = line.rstrip(" |")
            if line:
                lines.append(f"\u2022 {line}")
        if lines:
            return "\n".join(lines) + "\n"
        return ""

    def table_row(self, text: str) -> str:
        """Render a table row."""
        return text.rstrip(" |") + "\n"

    def table_cell(self, text: str, align: str | None = None, head: bool = False) -> str:
        """Render a table cell."""
        stripped = text.strip()
        if head:
            return f"<b>{stripped}</b> | "
        return f"{stripped} | "

    def finalize(self, data: str) -> str:
        """Strip trailing whitespace from the rendered output."""
        return data.strip()


_md_renderer = mistune.create_markdown(renderer=TelegramHTMLRenderer(), plugins=["table"])


def markdown_to_telegram_html(md_text: str) -> str:
    """Convert Markdown text to Telegram-compatible HTML."""
    return cast(str, _md_renderer(md_text))


# ---------------------------------------------------------------------------
# URL validation (SSRF protection)
# ---------------------------------------------------------------------------


class SSRFBlockedError(Exception):
    """Raised when a URL resolves to a private/internal address."""


def _is_ip_allowed(ip_str: str) -> bool:
    """Return False if the IP is private, loopback, link-local, reserved, or multicast."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return not (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved or addr.is_multicast)


def _validate_url(url: str) -> str | None:
    """Structural URL validation. Returns an error message string if invalid, None if OK."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return f"Invalid URL: {url}"

    if parsed.scheme not in ("http", "https"):
        return f"URL scheme '{parsed.scheme}' is not allowed. Only http and https are supported."

    hostname = parsed.hostname
    if not hostname:
        return f"Could not extract hostname from URL: {url}"

    return None


class _SSRFSafeTransport(httpx.AsyncHTTPTransport):
    """httpx transport that validates resolved IP addresses before connecting.

    This eliminates the DNS rebinding TOCTOU window by checking IPs at the
    socket level rather than in a separate pre-flight DNS lookup.
    """

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        hostname = request.url.host
        if hostname:
            addrinfo = await asyncio.to_thread(socket.getaddrinfo, hostname, None)
            for _family, _type, _proto, _canonname, sockaddr in addrinfo:
                ip = sockaddr[0]
                if not isinstance(ip, str):
                    raise SSRFBlockedError(f"URL resolves to an unsupported address format ({ip!r}). Access denied.")
                if not _is_ip_allowed(ip):
                    raise SSRFBlockedError(f"URL resolves to a private/internal address ({ip}). Access denied.")
        return await super().handle_async_request(request)


_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        transport = _SSRFSafeTransport(retries=1)
        _http_client = httpx.AsyncClient(
            transport=transport,
            follow_redirects=True,
            timeout=30.0,
            headers={"User-Agent": "Mozilla/5.0 (compatible; deeper-bot/1.0)"},
        )
    return _http_client


async def close_http_client() -> None:
    """Close the shared HTTP client. Call during application shutdown."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


# ---------------------------------------------------------------------------
# Tool argument models (Pydantic)
# ---------------------------------------------------------------------------


class WebSearchArgs(BaseModel):
    """Arguments for the web_search tool."""

    query: str = Field(min_length=1)
    max_results: int = Field(default=5, ge=1, le=15)


class WebFetchArgs(BaseModel):
    """Arguments for the web_fetch tool."""

    url: str = Field(min_length=1)


class AskUserArgs(BaseModel):
    """Arguments for the ask_user tool."""

    question: str = Field(min_length=1)


class FinishArgs(BaseModel):
    """Arguments for the finish tool."""

    result_markdown: str = Field(min_length=1)


class SetStatusArgs(BaseModel):
    """Arguments for the set_status tool."""

    todo_list: str = Field(min_length=1)


_TOOL_MODELS: dict[str, type[BaseModel]] = {
    "web_search": WebSearchArgs,
    "web_fetch": WebFetchArgs,
    "ask_user": AskUserArgs,
    "finish": FinishArgs,
    "set_status": SetStatusArgs,
}


def _validate_tool_args(name: str, args: dict) -> BaseModel | str:
    """Validate and parse tool arguments.

    Returns the parsed Pydantic model on success, or an error message string on failure.
    """
    model_cls = _TOOL_MODELS.get(name)
    if model_cls is None:
        return f"Unknown tool: {name}"
    try:
        return model_cls.model_validate(args)
    except ValidationError as e:
        errors = e.errors()
        details = "; ".join(f"{' -> '.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in errors)
        return f"Invalid arguments for {name}: {details}"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def _web_search(query: str, max_results: int = 5) -> str:
    # DDGS wraps requests.Session internally, which is not thread-safe.
    # A new instance per call is intentional for safe use with asyncio.to_thread.
    try:
        results = await asyncio.to_thread(DDGS().text, query, max_results=max_results)
    except Exception as e:
        logger.warning("web_search error for %r: %s", query, e)
        return f"Search failed: {e}"

    if not results:
        return "No results found."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        href = r.get("href", "")
        body = r.get("body", "")
        lines.append(f"{i}. {title}\n   URL: {href}\n   {body}")
    return "\n\n".join(lines)


async def _web_fetch(url: str, settings: Settings) -> str:
    validation_error = _validate_url(url)
    if validation_error:
        return validation_error

    client = _get_http_client()
    try:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > MAX_DOWNLOAD_SIZE:
                return f"Page too large to process ({int(content_length)} bytes, limit {MAX_DOWNLOAD_SIZE})."
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > MAX_DOWNLOAD_SIZE:
                    return f"Page too large to process (exceeded {MAX_DOWNLOAD_SIZE} bytes during download)."
                chunks.append(chunk)
            html = b"".join(chunks).decode("utf-8", errors="replace")
    except SSRFBlockedError as e:
        return str(e)
    except httpx.TimeoutException:
        return f"Request timed out for URL: {url}"
    except Exception as e:
        logger.warning("web_fetch download error for %s: %s", url, e)
        return f"Failed to fetch URL: {e}"

    if not html:
        return "Could not download content from this URL."

    try:
        content = await asyncio.to_thread(
            trafilatura.extract, html, output_format="markdown", include_links=True, include_tables=True
        )
    except Exception as e:
        logger.warning("web_fetch extraction error for %s: %s", url, e)
        return f"Failed to extract content: {e}"

    if not content:
        return "Could not extract meaningful content from this URL."

    if len(content) > MAX_FETCH_CONTENT_LENGTH:
        summary = await _summarize_web_content(content, settings)
        if summary:
            content = f"[Content summarized — original exceeded {len(content)} characters]\n\n" + summary
        else:
            content = content[:MAX_FETCH_CONTENT_LENGTH] + "\n\n[Content truncated]"
    return content


_WEB_SUMMARIZATION_PROMPT = (
    "Summarize the following web page content into a concise but thorough overview. "
    "Preserve: key facts, data points, statistics, arguments, conclusions, and source attributions. "
    "Omit navigation elements, boilerplate, and redundant information."
)


async def _summarize_web_content(content: str, settings: Settings) -> str | None:
    """Summarize long web content via LLM. Returns None on failure."""
    try:
        response = await llm_call_with_retry(
            {
                "model": settings.resolved_utility_model,
                "messages": [
                    {"role": "system", "content": _WEB_SUMMARIZATION_PROMPT},
                    {"role": "user", "content": content},
                ],
                "api_base": settings.llm_base_url,
                "api_key": settings.resolved_llm_api_key,
                "max_tokens": 3000,
            }
        )
        return response.choices[0].message.content or None
    except Exception as e:
        logger.warning("Web content summarization failed: %s", e)
        return None


async def _ask_user(question: str, session: Session, bot: Bot, chat_id: int) -> str:
    html_question = markdown_to_telegram_html(question)
    await bot.send_message(chat_id, html_question, parse_mode=ParseMode.HTML)

    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()
    session.set_awaiting_answer(future)

    try:
        answer = await asyncio.wait_for(future, timeout=3600)
    except TimeoutError:
        session.timeout_pending()
        with contextlib.suppress(Exception):
            await bot.send_message(
                chat_id,
                markdown_to_telegram_html(
                    "Response time expired. The research session will continue without your answer."
                ),
                parse_mode=ParseMode.HTML,
            )
        return "User did not respond within the time limit (60 minutes)."
    except asyncio.CancelledError:
        raise

    return answer


async def _finish(result_markdown: str, bot: Bot, chat_id: int) -> str:
    html_text = markdown_to_telegram_html(result_markdown)

    if len(html_text) <= MAX_TELEGRAM_INLINE_LENGTH:
        try:
            await bot.send_message(chat_id, html_text, parse_mode=ParseMode.HTML)
            return "Research delivered to user."
        except TelegramForbiddenError:
            raise
        except Exception as e:
            logger.warning("Failed to send HTML message, falling back to file: %s", e)

    # Send as .md file
    file_bytes = result_markdown.encode("utf-8")
    document = BufferedInputFile(file_bytes, filename="research_report.md")
    await bot.send_message(
        chat_id, markdown_to_telegram_html("Research complete. Full report attached below."), parse_mode=ParseMode.HTML
    )
    await bot.send_document(chat_id, document)
    return "Research delivered to user as file."


async def _set_status(todo_list: str, session: Session, bot: Bot, chat_id: int) -> str:
    session.todo_list = todo_list
    if not session._status_announced:
        full_md = f"TODO:\n\n{todo_list}\n\nUse /status to check progress."
        html = markdown_to_telegram_html(full_md)
        await bot.send_message(chat_id, html, parse_mode=ParseMode.HTML)
        session._status_announced = True
    return "Status updated."


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------


async def execute_tool(
    tool_call: ChatCompletionMessageToolCall,
    session: Session,
    bot: Bot,
    chat_id: int,
    settings: Settings,
) -> tuple[dict, bool]:
    """Execute a tool call and return (tool_result_message, is_finish)."""
    func = tool_call.function
    name = func.name
    if not name:
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": "Tool call missing name.",
        }, False

    try:
        args = json.loads(func.arguments)
    except json.JSONDecodeError, TypeError:
        args = {}

    is_finish = False

    # Validate arguments before dispatching
    validated = _validate_tool_args(name, args)
    if isinstance(validated, str):
        msg = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": validated,
        }
        return msg, False

    try:
        if name == "web_search":
            ws = cast(WebSearchArgs, validated)
            result = await _web_search(ws.query, ws.max_results)
        elif name == "web_fetch":
            wf = cast(WebFetchArgs, validated)
            result = await _web_fetch(wf.url, settings)
        elif name == "ask_user":
            au = cast(AskUserArgs, validated)
            result = await _ask_user(au.question, session, bot, chat_id)
        elif name == "finish":
            fi = cast(FinishArgs, validated)
            result = await _finish(fi.result_markdown, bot, chat_id)
            is_finish = True
        elif name == "set_status":
            ss = cast(SetStatusArgs, validated)
            result = await _set_status(ss.todo_list, session, bot, chat_id)
        else:
            result = f"Unknown tool: {name}"
    except asyncio.CancelledError:
        raise
    except TelegramForbiddenError:
        raise
    except Exception as e:
        logger.exception("Tool %s execution error", name)
        result = f"Tool execution failed: {e}"

    msg = {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result,
    }
    return msg, is_finish
