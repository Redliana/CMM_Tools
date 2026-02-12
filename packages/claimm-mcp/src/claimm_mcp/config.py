"""Configuration management for CLAIMM MCP Server."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # EDX API Configuration (Required)
    edx_api_key: str = Field(..., description="NETL EDX API Key")
    edx_base_url: str = Field(
        default="https://edx.netl.doe.gov/api/3/action",
        description="EDX API base URL",
    )

    # LLM Provider API Keys (at least one required)
    openai_api_key: str | None = Field(default=None, description="OpenAI API Key")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API Key")
    google_api_key: str | None = Field(default=None, description="Google AI API Key")
    xai_api_key: str | None = Field(default=None, description="xAI (Grok) API Key")

    # LLM Configuration
    default_llm_provider: Literal["openai", "anthropic", "google", "xai"] = Field(
        default="anthropic",
        description="Default LLM provider to use",
    )
    default_llm_model: str | None = Field(
        default=None,
        description="Default model to use (provider-specific). If not set, uses provider default.",
    )

    # CLAIMM-specific settings
    claimm_group: str = Field(
        default="claimm-mine-waste",
        description="EDX group identifier for CLAIMM data",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def get_llm_model(self) -> str:
        """Get the LLM model string for LiteLLM."""
        if self.default_llm_model:
            # If explicit model set, use provider prefix
            provider_prefix = {
                "openai": "",
                "anthropic": "anthropic/",
                "google": "gemini/",
                "xai": "xai/",
            }
            prefix = provider_prefix.get(self.default_llm_provider, "")
            return f"{prefix}{self.default_llm_model}"

        # Default models per provider
        defaults = {
            "openai": "gpt-4o",
            "anthropic": "anthropic/claude-sonnet-4-20250514",
            "google": "gemini/gemini-1.5-pro",
            "xai": "xai/grok-beta",
        }
        return defaults.get(self.default_llm_provider, "gpt-4o")

    def get_available_provider(self) -> str | None:
        """Get the first available LLM provider based on configured API keys."""
        if self.default_llm_provider == "openai" and self.openai_api_key:
            return "openai"
        if self.default_llm_provider == "anthropic" and self.anthropic_api_key:
            return "anthropic"
        if self.default_llm_provider == "google" and self.google_api_key:
            return "google"
        if self.default_llm_provider == "xai" and self.xai_api_key:
            return "xai"

        # Fallback to any available provider
        if self.openai_api_key:
            return "openai"
        if self.anthropic_api_key:
            return "anthropic"
        if self.google_api_key:
            return "google"
        if self.xai_api_key:
            return "xai"

        return None


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
