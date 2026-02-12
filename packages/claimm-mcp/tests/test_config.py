"""Tests for claimm_mcp.config module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from claimm_mcp.config import Settings, get_settings


class TestSettingsConstruction:
    """Tests for building Settings from environment variables."""

    def test_settings_with_required_env_vars(self, env_vars: dict[str, str]) -> None:
        """Settings should construct when EDX_API_KEY is provided."""
        s = Settings()  # type: ignore[call-arg]
        assert s.edx_api_key == env_vars["EDX_API_KEY"]

    def test_settings_missing_edx_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings should raise ValidationError when EDX_API_KEY is missing."""
        monkeypatch.delenv("EDX_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Ensure no .env file pollutes the test
        monkeypatch.setattr(
            "pydantic_settings.BaseSettings.__init_subclass__", lambda **kw: None, raising=False
        )
        with pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]

    def test_settings_default_provider_is_anthropic(self, settings: Settings) -> None:
        """Default LLM provider should be anthropic."""
        assert settings.default_llm_provider == "anthropic"

    def test_settings_default_base_url(self, settings: Settings) -> None:
        """Default EDX base URL should be the NETL endpoint."""
        assert settings.edx_base_url == "https://edx.netl.doe.gov/api/3/action"

    def test_settings_default_claimm_group(self, settings: Settings) -> None:
        """Default CLAIMM group should be 'claimm-mine-waste'."""
        assert settings.claimm_group == "claimm-mine-waste"

    def test_settings_llm_keys_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings should pick up all LLM provider keys from env."""
        monkeypatch.setenv("EDX_API_KEY", "edx-key")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
        monkeypatch.setenv("XAI_API_KEY", "xai-key")

        s = Settings()  # type: ignore[call-arg]
        assert s.openai_api_key == "openai-key"
        assert s.anthropic_api_key == "anthropic-key"
        assert s.google_api_key == "google-key"
        assert s.xai_api_key == "xai-key"

    def test_settings_no_llm_keys_defaults_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM keys should be None when not set in environment."""
        monkeypatch.setenv("EDX_API_KEY", "edx-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)

        s = Settings()  # type: ignore[call-arg]
        assert s.openai_api_key is None
        assert s.anthropic_api_key is None
        assert s.google_api_key is None
        assert s.xai_api_key is None

    def test_settings_custom_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings should accept a custom default_llm_provider."""
        monkeypatch.setenv("EDX_API_KEY", "edx-key")
        monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

        s = Settings()  # type: ignore[call-arg]
        assert s.default_llm_provider == "openai"


class TestGetLlmModel:
    """Tests for Settings.get_llm_model()."""

    def test_default_model_for_anthropic(self, settings: Settings) -> None:
        """Anthropic default model should include the 'anthropic/' prefix."""
        model = settings.get_llm_model()
        assert model.startswith("anthropic/")
        assert "claude" in model

    @pytest.mark.parametrize(
        ("provider", "expected_prefix"),
        [
            ("openai", ""),
            ("anthropic", "anthropic/"),
            ("google", "gemini/"),
            ("xai", "xai/"),
        ],
    )
    def test_explicit_model_uses_provider_prefix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider: str,
        expected_prefix: str,
    ) -> None:
        """When default_llm_model is explicitly set, it should be prefixed by provider."""
        monkeypatch.setenv("EDX_API_KEY", "edx-key")
        monkeypatch.setenv("DEFAULT_LLM_PROVIDER", provider)
        monkeypatch.setenv("DEFAULT_LLM_MODEL", "my-custom-model")

        s = Settings()  # type: ignore[call-arg]
        result = s.get_llm_model()
        assert result == f"{expected_prefix}my-custom-model"

    def test_default_model_per_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each provider should have a sensible default model when none is explicitly set."""
        expected_defaults = {
            "openai": "gpt-4o",
            "anthropic": "anthropic/claude-sonnet-4-20250514",
            "google": "gemini/gemini-1.5-pro",
            "xai": "xai/grok-beta",
        }
        for provider, expected_model in expected_defaults.items():
            monkeypatch.setenv("EDX_API_KEY", "edx-key")
            monkeypatch.setenv("DEFAULT_LLM_PROVIDER", provider)
            monkeypatch.delenv("DEFAULT_LLM_MODEL", raising=False)

            s = Settings()  # type: ignore[call-arg]
            assert s.get_llm_model() == expected_model, (
                f"Default model for {provider} should be {expected_model}"
            )


class TestGetAvailableProvider:
    """Tests for Settings.get_available_provider()."""

    def test_returns_preferred_when_key_set(self, settings: Settings) -> None:
        """Should return the default provider when its API key is available."""
        # settings fixture has ANTHROPIC_API_KEY set and default_llm_provider=anthropic
        assert settings.get_available_provider() == "anthropic"

    def test_falls_back_to_any_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When preferred provider has no key, should fall back to another."""
        monkeypatch.setenv("EDX_API_KEY", "edx-key")
        monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
        monkeypatch.delenv("XAI_API_KEY", raising=False)

        s = Settings()  # type: ignore[call-arg]
        assert s.get_available_provider() == "google"

    def test_returns_none_when_no_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return None when no LLM provider API keys are set."""
        monkeypatch.setenv("EDX_API_KEY", "edx-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)

        s = Settings()  # type: ignore[call-arg]
        assert s.get_available_provider() is None

    @pytest.mark.parametrize(
        ("provider_env", "key_env", "key_val", "expected"),
        [
            ("openai", "OPENAI_API_KEY", "oai-key", "openai"),
            ("anthropic", "ANTHROPIC_API_KEY", "ant-key", "anthropic"),
            ("google", "GOOGLE_API_KEY", "ggl-key", "google"),
            ("xai", "XAI_API_KEY", "xai-key", "xai"),
        ],
    )
    def test_each_provider_returned_when_key_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider_env: str,
        key_env: str,
        key_val: str,
        expected: str,
    ) -> None:
        """Each provider should be returned when it is the default and its key is set."""
        monkeypatch.setenv("EDX_API_KEY", "edx-key")
        monkeypatch.setenv("DEFAULT_LLM_PROVIDER", provider_env)
        # Clear all keys first
        for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        # Set only the one we want
        monkeypatch.setenv(key_env, key_val)

        s = Settings()  # type: ignore[call-arg]
        assert s.get_available_provider() == expected


class TestGetSettingsSingleton:
    """Tests for the get_settings() module-level function."""

    def test_get_settings_returns_settings_instance(self, settings_with_reset: Settings) -> None:
        """get_settings should return the globally cached Settings."""
        result = get_settings()
        assert isinstance(result, Settings)
        assert result is settings_with_reset

    def test_get_settings_caches_instance(self, settings_with_reset: Settings) -> None:
        """Calling get_settings twice should return the same object."""
        first = get_settings()
        second = get_settings()
        assert first is second
