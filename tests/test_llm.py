from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

from deeper_bot.llm import llm_call_with_retry


class TestLlmCallWithRetry:
    async def test_succeeds_first_try(self):
        mock_response = MagicMock()
        with patch("deeper_bot.llm.litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await llm_call_with_retry({"model": "test"})
        assert result is mock_response

    async def test_retries_on_rate_limit_then_succeeds(self):
        mock_response = MagicMock()
        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise litellm.RateLimitError("rate limited", "test", "test")
            return mock_response

        with (
            patch("deeper_bot.llm.litellm.acompletion", side_effect=side_effect),
            patch("deeper_bot.llm.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await llm_call_with_retry({"model": "test"})
        assert result is mock_response
        assert call_count == 3

    async def test_exhausts_retries(self):
        async def side_effect(**kwargs):
            raise litellm.RateLimitError("rate limited", "test", "test")

        with (
            patch("deeper_bot.llm.litellm.acompletion", side_effect=side_effect),
            patch("deeper_bot.llm.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(litellm.RateLimitError),
        ):
            await llm_call_with_retry({"model": "test"})

    async def test_non_retryable_error_fails_immediately(self):
        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            raise litellm.AuthenticationError("bad key", "test", "test")

        with (
            patch("deeper_bot.llm.litellm.acompletion", side_effect=side_effect),
            pytest.raises(litellm.AuthenticationError),
        ):
            await llm_call_with_retry({"model": "test"})
        assert call_count == 1

    async def test_context_window_not_retried(self):
        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            raise litellm.ContextWindowExceededError("too long", "test", "test")

        with (
            patch("deeper_bot.llm.litellm.acompletion", side_effect=side_effect),
            pytest.raises(litellm.ContextWindowExceededError),
        ):
            await llm_call_with_retry({"model": "test"})
        assert call_count == 1
