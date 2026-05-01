"""LLM client wrapper with retry logic for transient errors."""

import asyncio
import logging

import litellm

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
