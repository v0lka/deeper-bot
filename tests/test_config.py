import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from deeper_bot.config import Settings


class _TestSettings(Settings):
    """Settings subclass that ignores .env files for test isolation."""

    model_config = SettingsConfigDict(env_file=None, env_file_encoding="utf-8")


def _make_settings(**kwargs):
    """Create Settings with env isolation -- clears env vars and ignores .env file.

    Values are passed via env vars by default.  For complex fields like
    ``allowed_users`` where the raw string must reach the pydantic validator
    *before* pydantic-settings attempts JSON decoding, pass them in
    ``_init_kwargs`` instead.
    """
    init_kwargs = kwargs.pop("_init_kwargs", {})
    env = {}
    for key in list(kwargs):
        env[key.upper()] = str(kwargs.pop(key))
    with patch.dict(os.environ, env, clear=True):
        return _TestSettings(**init_kwargs)


class TestSettings:
    def test_empty_bot_token_rejected(self):
        with pytest.raises(ValidationError, match="BOT_TOKEN must be set"):
            _make_settings(bot_token="", llm_base_url="http://x", llm_model="m", llm_api_key="sk-test")

    def test_whitespace_bot_token_rejected(self):
        with pytest.raises(ValidationError, match="BOT_TOKEN must be set"):
            _make_settings(bot_token="   ", llm_base_url="http://x", llm_model="m", llm_api_key="sk-test")

    def test_valid_bot_token_accepted(self):
        s = _make_settings(bot_token="123:ABC", llm_base_url="http://x", llm_model="m", llm_api_key="sk-test")
        assert s.bot_token == "123:ABC"

    def test_llm_api_key_direct_value(self):
        s = _make_settings(
            bot_token="123:ABC",
            llm_base_url="http://x",
            llm_model="m",
            llm_api_key="sk-direct",
        )
        assert s.resolved_llm_api_key == "sk-direct"

    def test_llm_api_key_env_reference_missing(self):
        s = _make_settings(
            bot_token="123:ABC",
            llm_base_url="http://x",
            llm_model="m",
            llm_api_key="${NONEXISTENT_VAR_XYZ}",
        )
        with pytest.raises(RuntimeError, match="NONEXISTENT_VAR_XYZ"):
            _ = s.resolved_llm_api_key

    def test_llm_api_key_env_reference_present(self):
        s = _make_settings(
            bot_token="123:ABC",
            llm_base_url="http://x",
            llm_model="m",
            llm_api_key="${MY_TEST_KEY}",
        )
        with patch.dict(os.environ, {"MY_TEST_KEY": "sk-secret"}):
            assert s.resolved_llm_api_key == "sk-secret"

    def test_parse_allowed_users_empty_string(self):
        s = _make_settings(
            bot_token="123:ABC",
            llm_base_url="http://x",
            llm_model="m",
            llm_api_key="sk-test",
            _init_kwargs={"allowed_users": ""},
        )
        assert s.allowed_users == []

    def test_parse_allowed_users_csv(self):
        s = _make_settings(
            bot_token="123:ABC",
            llm_base_url="http://x",
            llm_model="m",
            llm_api_key="sk-test",
            _init_kwargs={"allowed_users": "111, 222, 333"},
        )
        assert s.allowed_users == [111, 222, 333]

    def test_parse_allowed_users_list(self):
        s = _make_settings(
            bot_token="123:ABC",
            llm_base_url="http://x",
            llm_model="m",
            llm_api_key="sk-test",
            allowed_users="[1, 2]",
        )
        assert s.allowed_users == [1, 2]
