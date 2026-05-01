from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

from deeper_bot.llm import build_llm_kwargs, llm_call_with_retry


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


# ---------------------------------------------------------------------------
# build_llm_kwargs tests
# ---------------------------------------------------------------------------


class TestBuildLlmKwargs:
    def _make_settings(self, *, use_reasoning: bool = False, reasoning_effort: str = "high"):
        s = MagicMock()
        s.llm_model = "primary-model"
        s.llm_base_url = "http://localhost"
        s.resolved_llm_api_key = "sk-test"
        s.llm_use_reasoning = use_reasoning
        s.llm_reasoning_effort = reasoning_effort
        s.llm_temperature = 0.6
        s.llm_utility_temperature = 0.2
        s.resolved_utility_model = "utility-model"
        return s

    def test_default_model_uses_primary(self):
        settings = self._make_settings()
        result = build_llm_kwargs(settings, messages=[{"role": "user", "content": "hi"}])
        assert result["model"] == "primary-model"

    def test_model_override(self):
        settings = self._make_settings()
        result = build_llm_kwargs(settings, model="custom-model", messages=[])
        assert result["model"] == "custom-model"

    def test_api_base_and_key_populated(self):
        settings = self._make_settings()
        result = build_llm_kwargs(settings, messages=[])
        assert result["api_base"] == "http://localhost"
        assert result["api_key"] == "sk-test"

    def test_reasoning_effort_added_when_enabled(self):
        settings = self._make_settings(use_reasoning=True, reasoning_effort="medium")
        result = build_llm_kwargs(settings, messages=[])
        assert result["reasoning_effort"] == "medium"

    def test_reasoning_effort_suppressed_with_max_tokens(self):
        settings = self._make_settings(use_reasoning=True)
        result = build_llm_kwargs(settings, messages=[], max_tokens=1000)
        assert "reasoning_effort" not in result
        assert result["max_tokens"] == 1000

    def test_reasoning_effort_absent_when_disabled(self):
        settings = self._make_settings(use_reasoning=False)
        result = build_llm_kwargs(settings, messages=[])
        assert "reasoning_effort" not in result

    def test_overrides_merged(self):
        settings = self._make_settings()
        tools = [{"type": "function", "function": {"name": "test"}}]
        result = build_llm_kwargs(settings, messages=[], tools=tools)
        assert result["tools"] is tools

    def test_messages_passed_through(self):
        settings = self._make_settings()
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        result = build_llm_kwargs(settings, messages=msgs)
        assert result["messages"] is msgs

    def test_default_temperature_included(self):
        settings = self._make_settings()
        result = build_llm_kwargs(settings, messages=[])
        assert result["temperature"] == 0.6

    def test_temperature_override(self):
        settings = self._make_settings()
        result = build_llm_kwargs(settings, messages=[], temperature=0.9)
        assert result["temperature"] == 0.9

    def test_temperature_excluded_when_reasoning_active(self):
        settings = self._make_settings(use_reasoning=True)
        result = build_llm_kwargs(settings, messages=[])
        assert "temperature" not in result
        assert result["reasoning_effort"] == "high"

    def test_temperature_included_when_reasoning_disabled(self):
        settings = self._make_settings(use_reasoning=False)
        result = build_llm_kwargs(settings, messages=[])
        assert result["temperature"] == 0.6
        assert "reasoning_effort" not in result

    def test_temperature_included_with_max_tokens(self):
        settings = self._make_settings(use_reasoning=True)
        result = build_llm_kwargs(settings, messages=[], max_tokens=500)
        assert result["temperature"] == 0.6
        assert "reasoning_effort" not in result
