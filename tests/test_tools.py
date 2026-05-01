import json
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deeper_bot.session import Session, SessionState
from deeper_bot.tools import (
    SSRFBlockedError,
    TelegramHTMLRenderer,
    _is_ip_allowed,
    _validate_url,
    execute_tool,
    markdown_to_telegram_html,
)

# ---------------------------------------------------------------------------
# TelegramHTMLRenderer tests
# ---------------------------------------------------------------------------


class TestTelegramHTMLRenderer:
    def test_bold(self):
        html = markdown_to_telegram_html("**bold**")
        assert "<b>bold</b>" in html

    def test_italic(self):
        html = markdown_to_telegram_html("*italic*")
        assert "<i>italic</i>" in html

    def test_heading(self):
        html = markdown_to_telegram_html("# Title")
        assert "<b>Title</b>" in html

    def test_code_block(self):
        html = markdown_to_telegram_html("```python\nprint(1)\n```")
        assert "<pre>" in html
        assert "print(1)" in html

    def test_inline_code(self):
        html = markdown_to_telegram_html("Use `foo()` here")
        assert "<code>foo()</code>" in html

    def test_blockquote(self):
        html = markdown_to_telegram_html("> quoted text")
        assert "<blockquote>" in html

    def test_link(self):
        html = markdown_to_telegram_html("[click](http://example.com)")
        assert '<a href="http://example.com">click</a>' in html

    def test_unordered_list_has_bullets(self):
        html = markdown_to_telegram_html("- item one\n- item two\n- item three")
        assert "\u2022 item one" in html
        assert "\u2022 item two" in html
        assert "\u2022 item three" in html

    def test_ordered_list_has_numbers(self):
        html = markdown_to_telegram_html("1. first\n2. second\n3. third")
        assert "1. first" in html
        assert "2. second" in html
        assert "3. third" in html

    def test_nested_lists(self):
        md = "- outer\n  - inner a\n  - inner b\n- outer2"
        html = markdown_to_telegram_html(md)
        assert "\u2022 outer" in html
        assert "\u2022 inner a" in html
        assert "\u2022 outer2" in html

    def test_image_renders_as_link(self):
        html = markdown_to_telegram_html("![alt](http://example.com/img.png)")
        assert "<a href=" in html
        assert "alt" in html

    def test_thematic_break(self):
        html = markdown_to_telegram_html("---")
        # Should not crash, renders as newline
        assert isinstance(html, str)

    def test_table(self):
        html = markdown_to_telegram_html("| a | b |\n|---|---|\n| 1 | 2 |")
        assert "<b>a</b> | <b>b</b>" in html
        assert "\u2022 1 | 2" in html

    def test_finalize_strips(self):
        r = TelegramHTMLRenderer()
        assert r.finalize("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# URL validation tests
# ---------------------------------------------------------------------------


class TestValidateUrl:
    """Structural URL validation (scheme, hostname)."""

    def test_rejects_file_scheme(self):
        err = _validate_url("file:///etc/passwd")
        assert err is not None
        assert "not allowed" in err

    def test_rejects_ftp_scheme(self):
        err = _validate_url("ftp://example.com/file")
        assert err is not None
        assert "not allowed" in err

    def test_rejects_javascript_scheme(self):
        err = _validate_url("javascript:alert(1)")
        assert err is not None
        assert "not allowed" in err

    def test_accepts_http(self):
        assert _validate_url("http://example.com") is None

    def test_accepts_https(self):
        assert _validate_url("https://example.com") is None

    def test_no_hostname(self):
        err = _validate_url("http://")
        assert err is not None
        assert "hostname" in err.lower()


class TestIsIpAllowed:
    """IP-level SSRF checks."""

    def test_public_ipv4_allowed(self):
        assert _is_ip_allowed("93.184.216.34") is True

    def test_loopback_blocked(self):
        assert _is_ip_allowed("127.0.0.1") is False

    def test_private_10x_blocked(self):
        assert _is_ip_allowed("10.0.0.1") is False

    def test_private_172_blocked(self):
        assert _is_ip_allowed("172.16.0.1") is False

    def test_private_192_blocked(self):
        assert _is_ip_allowed("192.168.1.1") is False

    def test_link_local_blocked(self):
        assert _is_ip_allowed("169.254.169.254") is False

    def test_ipv6_loopback_blocked(self):
        assert _is_ip_allowed("::1") is False

    def test_multicast_blocked(self):
        assert _is_ip_allowed("224.0.0.1") is False

    def test_invalid_ip_blocked(self):
        assert _is_ip_allowed("not-an-ip") is False


class TestSSRFTransport:
    """Transport-level SSRF blocking prevents DNS rebinding."""

    async def test_blocks_private_ip_at_connect(self):
        from deeper_bot.tools import _SSRFSafeTransport

        transport = _SSRFSafeTransport()
        request = MagicMock()
        request.url.host = "evil.example.com"

        with (
            patch(
                "deeper_bot.tools.socket.getaddrinfo",
                return_value=[(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))],
            ),
            pytest.raises(SSRFBlockedError, match="private"),
        ):
            await transport.handle_async_request(request)

    async def test_allows_public_ip(self):
        from deeper_bot.tools import _SSRFSafeTransport

        transport = _SSRFSafeTransport()
        request = MagicMock()
        request.url.host = "example.com"

        mock_response = MagicMock()
        with (
            patch(
                "deeper_bot.tools.socket.getaddrinfo",
                return_value=[(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))],
            ),
            patch.object(
                transport.__class__.__bases__[0],
                "handle_async_request",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
        ):
            result = await transport.handle_async_request(request)
        assert result is mock_response


# ---------------------------------------------------------------------------
# execute_tool dispatch tests
# ---------------------------------------------------------------------------


def _make_tool_call(name: str, arguments: dict, call_id: str = "call_1"):
    """Create a mock tool call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def _make_settings():
    """Create a mock Settings object for tool execution."""
    s = MagicMock()
    s.llm_model = "test-model"
    s.llm_base_url = "http://localhost"
    s.llm_api_key = "test-key"
    return s


class TestExecuteTool:
    @pytest.fixture
    def session(self):
        return Session(chat_id=123)

    @pytest.fixture
    def bot(self):
        b = AsyncMock()
        b.send_message = AsyncMock()
        b.send_document = AsyncMock()
        return b

    @pytest.fixture
    def settings(self):
        return _make_settings()

    async def test_unknown_tool(self, session, bot, settings):
        tc = _make_tool_call("nonexistent", {})
        msg, is_finish = await execute_tool(tc, session, bot, 123, settings)
        assert msg["content"] == "Unknown tool: nonexistent"
        assert not is_finish

    async def test_malformed_json_arguments(self, session, bot, settings):
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "web_search"
        tc.function.arguments = "not json{{"
        msg, is_finish = await execute_tool(tc, session, bot, 123, settings)
        # Should hit validation error (empty args -> query missing)
        assert "Invalid arguments" in msg["content"]

    async def test_none_arguments_handled(self, session, bot, settings):
        """json.loads(None) raises TypeError — must be caught, not propagated."""
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "web_search"
        tc.function.arguments = None
        msg, is_finish = await execute_tool(tc, session, bot, 123, settings)
        assert "Invalid arguments" in msg["content"]
        assert not is_finish

    async def test_web_search_invalid_query_type(self, session, bot, settings):
        tc = _make_tool_call("web_search", {"query": 123})
        msg, is_finish = await execute_tool(tc, session, bot, 123, settings)
        assert "Invalid arguments" in msg["content"]
        assert not is_finish

    async def test_web_search_empty_query(self, session, bot, settings):
        tc = _make_tool_call("web_search", {"query": ""})
        msg, is_finish = await execute_tool(tc, session, bot, 123, settings)
        assert "Invalid arguments" in msg["content"]

    async def test_web_search_max_results_too_high(self, session, bot, settings):
        tc = _make_tool_call("web_search", {"query": "test", "max_results": 20})
        msg, is_finish = await execute_tool(tc, session, bot, 123, settings)
        assert "Invalid arguments" in msg["content"]
        assert not is_finish

    async def test_web_fetch_invalid_url(self, session, bot, settings):
        tc = _make_tool_call("web_fetch", {"url": ""})
        msg, is_finish = await execute_tool(tc, session, bot, 123, settings)
        assert "Invalid arguments" in msg["content"]

    async def test_ask_user_invalid_question(self, session, bot, settings):
        tc = _make_tool_call("ask_user", {"question": ""})
        msg, is_finish = await execute_tool(tc, session, bot, 123, settings)
        assert "Invalid arguments" in msg["content"]

    async def test_finish_invalid_markdown(self, session, bot, settings):
        tc = _make_tool_call("finish", {"result_markdown": ""})
        msg, is_finish = await execute_tool(tc, session, bot, 123, settings)
        assert "Invalid arguments" in msg["content"]

    async def test_web_search_valid(self, session, bot, settings):
        tc = _make_tool_call("web_search", {"query": "test query"})
        with patch("deeper_bot.tools.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text.return_value = [
                {"title": "Result", "href": "http://example.com", "body": "snippet"}
            ]
            msg, is_finish = await execute_tool(tc, session, bot, 123, settings)
        assert "Result" in msg["content"]
        assert not is_finish

    async def test_finish_valid_small(self, session, bot, settings):
        tc = _make_tool_call("finish", {"result_markdown": "# Report\nDone."})
        msg, is_finish = await execute_tool(tc, session, bot, 123, settings)
        assert is_finish
        assert "delivered" in msg["content"].lower()
        bot.send_message.assert_called()


# ---------------------------------------------------------------------------
# ask_user timeout state fix test
# ---------------------------------------------------------------------------


class TestAskUserTimeout:
    async def test_timeout_resets_state_to_researching(self):
        """After ask_user timeout, state should be RESEARCHING, not AWAITING_ANSWER."""
        from deeper_bot.tools import _ask_user

        session = Session(chat_id=1)
        bot = AsyncMock()
        bot.send_message = AsyncMock()

        # Patch wait_for to immediately raise TimeoutError
        with patch("deeper_bot.tools.asyncio.wait_for", side_effect=TimeoutError):
            result = await _ask_user("question?", session, bot, 1)

        assert session.state == SessionState.RESEARCHING
        assert session._pending_future is None
        assert "time limit" in result.lower()
        # Verify user was notified about the timeout
        timeout_calls = [call for call in bot.send_message.call_args_list if "expired" in str(call).lower()]
        assert len(timeout_calls) == 1


# ---------------------------------------------------------------------------
# set_status tool tests
# ---------------------------------------------------------------------------


class TestSetStatus:
    @pytest.fixture
    def session(self):
        return Session(chat_id=123)

    @pytest.fixture
    def bot(self):
        b = AsyncMock()
        b.send_message = AsyncMock()
        return b

    @pytest.fixture
    def settings(self):
        return _make_settings()

    async def test_set_status_stores_todo_list(self, session, bot):
        from deeper_bot.tools import _set_status

        await _set_status("- [ ] Step 1\n- [ ] Step 2", session, bot, 123)
        assert session.todo_list == "- [ ] Step 1\n- [ ] Step 2"

    async def test_set_status_first_call_sends_message(self, session, bot):
        from deeper_bot.tools import _set_status

        assert not session._status_announced
        await _set_status("- [ ] Step 1", session, bot, 123)
        assert session._status_announced
        bot.send_message.assert_called_once()
        sent_text = bot.send_message.call_args.args[1]
        assert "/status" in sent_text

    async def test_set_status_subsequent_calls_silent(self, session, bot):
        from deeper_bot.tools import _set_status

        session._status_announced = True
        await _set_status("- [x] Step 1\n- [ ] Step 2", session, bot, 123)
        assert session.todo_list == "- [x] Step 1\n- [ ] Step 2"
        bot.send_message.assert_not_called()

    async def test_set_status_returns_confirmation(self, session, bot):
        from deeper_bot.tools import _set_status

        result = await _set_status("- [ ] Step 1", session, bot, 123)
        assert result == "Status updated."

    async def test_execute_tool_set_status(self, session, bot, settings):
        tc = _make_tool_call("set_status", {"todo_list": "- [ ] Research"})
        msg, is_finish = await execute_tool(tc, session, bot, 123, settings)
        assert not is_finish
        assert "updated" in msg["content"].lower()
        assert session.todo_list == "- [ ] Research"

    async def test_set_status_empty_rejected(self, session, bot, settings):
        tc = _make_tool_call("set_status", {"todo_list": ""})
        msg, is_finish = await execute_tool(tc, session, bot, 123, settings)
        assert "Invalid arguments" in msg["content"]
        assert not is_finish


# ---------------------------------------------------------------------------
# web_fetch summarization tests
# ---------------------------------------------------------------------------


class TestWebFetchSummarization:
    def _mock_http_stream(self, html_text="<html>ok</html>", status_code=200):
        """Return a context manager that patches _get_http_client for streaming."""
        mock_response = AsyncMock()
        mock_response.status_code = status_code
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {}

        async def aiter_bytes():
            yield html_text.encode("utf-8")

        mock_response.aiter_bytes = aiter_bytes
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        return patch("deeper_bot.tools._get_http_client", return_value=mock_client)

    async def test_short_content_returned_as_is(self):
        """Content under the limit should be returned without summarization."""
        from deeper_bot.tools import _web_fetch

        short_content = "Short article content."
        settings = _make_settings()

        with (
            self._mock_http_stream(),
            patch("deeper_bot.tools.trafilatura.extract", return_value=short_content),
            patch("deeper_bot.tools.llm_call_with_retry", new_callable=AsyncMock) as mock_llm,
        ):
            result = await _web_fetch("http://example.com", settings)

        assert result == short_content
        mock_llm.assert_not_called()

    async def test_long_content_summarized(self):
        """Content over the limit should be summarized via LLM."""
        from deeper_bot.tools import _web_fetch

        long_content = "x" * 20_000
        summary_text = "This is a summary of the long content."
        settings = _make_settings()

        mock_llm_response = MagicMock()
        mock_llm_response.choices = [MagicMock(message=MagicMock(content=summary_text))]

        with (
            self._mock_http_stream(),
            patch("deeper_bot.tools.trafilatura.extract", return_value=long_content),
            patch("deeper_bot.tools.llm_call_with_retry", new_callable=AsyncMock, return_value=mock_llm_response),
        ):
            result = await _web_fetch("http://example.com", settings)

        assert result.startswith("[Content summarized")
        assert "20000" in result
        assert summary_text in result

    async def test_summarization_failure_falls_back_to_truncation(self):
        """If LLM summarization fails, should fall back to truncation."""
        from deeper_bot.tools import _web_fetch

        long_content = "y" * 20_000
        settings = _make_settings()

        with (
            self._mock_http_stream(),
            patch("deeper_bot.tools.trafilatura.extract", return_value=long_content),
            patch("deeper_bot.tools.llm_call_with_retry", new_callable=AsyncMock, side_effect=Exception("LLM error")),
        ):
            result = await _web_fetch("http://example.com", settings)

        assert result.endswith("[Content truncated]")
        assert len(result) < len(long_content)

    async def test_ssrf_blocked_returns_error_message(self):
        """SSRF-blocked URL should return error string, not raise."""
        from deeper_bot.tools import SSRFBlockedError, _web_fetch

        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(
            side_effect=SSRFBlockedError("private address (127.0.0.1). Access denied.")
        )
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        settings = _make_settings()

        with patch("deeper_bot.tools._get_http_client", return_value=mock_client):
            result = await _web_fetch("http://evil.com", settings)

        assert "Access denied" in result

    async def test_timeout_returns_error_message(self):
        """httpx timeout should return a user-friendly error."""
        import httpx

        from deeper_bot.tools import _web_fetch

        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        settings = _make_settings()

        with patch("deeper_bot.tools._get_http_client", return_value=mock_client):
            result = await _web_fetch("http://slow.example.com", settings)

        assert "timed out" in result.lower()
