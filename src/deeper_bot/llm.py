"""LLM client wrapper with retry logic for transient errors."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import litellm

if TYPE_CHECKING:
    from deeper_bot.config import Settings

logger = logging.getLogger(__name__)

MAX_LLM_ATTEMPTS = 3
_RETRY_DELAYS = (2, 4)
RETRYABLE_ERRORS = (
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
    litellm.APIConnectionError,
    litellm.Timeout,
    litellm.InternalServerError,
)


async def llm_call_with_retry(kwargs: dict) -> litellm.ModelResponse:
    """Call litellm.acompletion with retry for transient errors."""
    last_exc: Exception | None = None
    for attempt in range(MAX_LLM_ATTEMPTS):
        try:
            return await litellm.acompletion(**kwargs)
        except RETRYABLE_ERRORS as e:
            last_exc = e
            if attempt < MAX_LLM_ATTEMPTS - 1:
                delay = _RETRY_DELAYS[attempt]
                logger.warning(
                    "LLM call attempt %d/%d failed (%s), retrying in %ds",
                    attempt + 1,
                    MAX_LLM_ATTEMPTS,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.warning("LLM call failed after %d attempts: %s", MAX_LLM_ATTEMPTS, e)
    raise last_exc  # type: ignore[misc]


def build_llm_kwargs(
    settings: Settings,
    *,
    model: str | None = None,
    messages: list[dict[str, Any]],
    temperature: float | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a kwargs dict for ``llm_call_with_retry``.

    Args:
        settings: Application settings providing model, base URL, and API key.
        model: Override the model name (defaults to ``settings.llm_model``).
        messages: The message list for the LLM call.
        temperature: Override the temperature (defaults to ``settings.llm_temperature``).
        **overrides: Extra kwargs merged into the result (e.g. ``tools``, ``max_tokens``).
    """
    kwargs: dict[str, Any] = {
        "model": model or settings.llm_model,
        "messages": messages,
        "api_base": settings.llm_base_url,
        "api_key": settings.resolved_llm_api_key,
    }
    # Reasoning effort is only applicable for primary-model agent calls,
    # not utility summarization calls which set max_tokens.
    will_use_reasoning = settings.llm_use_reasoning and "max_tokens" not in overrides
    if will_use_reasoning:
        kwargs["reasoning_effort"] = settings.llm_reasoning_effort
    else:
        # Reasoning models (o1/o3) do not support the temperature parameter.
        kwargs["temperature"] = temperature if temperature is not None else settings.llm_temperature
    kwargs.update(overrides)
    return kwargs
