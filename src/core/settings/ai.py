# --- @ai START ---
# model: gpt-5
# tool: codex-cli
# role: architectural-refactor
# reviewed: no
# date: 2026-04-02
# --- @ai END ---
from __future__ import annotations

from typing import Any, Mapping

from core.secure_storage import load_secret, save_secret

from .base import SettingsGroupBase


class AISettings(SettingsGroupBase):
    _VALID_THINKING_MODES = {"off", "standard", "high"}

    def __init__(self, manager, *, secret_service: str) -> None:
        super().__init__(manager)
        self._secret_service = str(secret_service)

    def get_llm_provider(self) -> str:
        value = self._value("llm_provider", "openai")
        return str(value) if value else "openai"

    def set_llm_provider(self, provider: str) -> None:
        self._set_value("llm_provider", provider or "openai")

    def _model_key(self, provider: str) -> str:
        return f"llm_model_{provider or 'openai'}"

    def get_llm_model(self, provider: str) -> str:
        stored = self._value(self._model_key(provider), "")
        if stored:
            return str(stored)
        return self.default_llm_model(provider)

    def set_llm_model(self, model: str, provider: str) -> None:
        self._set_value(self._model_key(provider), model)

    @staticmethod
    def default_llm_model(provider: str) -> str:
        return "gpt-4o-mini" if provider == "openai" else "llama3.2"

    def get_llm_thinking_mode(self) -> str:
        value = self._value("llm_thinking_mode", "standard")
        mode = str(value or "standard").lower()
        return mode if mode in self._VALID_THINKING_MODES else "standard"

    def set_llm_thinking_mode(self, mode: str) -> None:
        normalized = str(mode or "standard").lower()
        if normalized not in self._VALID_THINKING_MODES:
            normalized = "standard"
        self._set_value("llm_thinking_mode", normalized)

    def get_openai_api_key(self) -> str:
        return load_secret(self._secret_service, "openai_api_key") or ""

    def set_openai_api_key(self, api_key: str | None) -> None:
        save_secret(self._secret_service, "openai_api_key", api_key)

    def defaults(self) -> dict[str, Any]:
        return {
            "provider": "openai",
            "models": {
                "openai": self.default_llm_model("openai"),
                "ollama": self.default_llm_model("ollama"),
            },
            "thinking_mode": "standard",
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.get_llm_provider(),
            "models": {
                "openai": self.get_llm_model("openai"),
                "ollama": self.get_llm_model("ollama"),
            },
            "thinking_mode": self.get_llm_thinking_mode(),
        }

    def update_from_dict(self, payload: Mapping[str, Any]) -> None:
        if "provider" in payload:
            self.set_llm_provider(str(payload.get("provider") or "openai"))
        models = payload.get("models")
        if isinstance(models, Mapping):
            for provider in ("openai", "ollama"):
                if provider in models:
                    self.set_llm_model(str(models.get(provider) or ""), provider)
        if "thinking_mode" in payload:
            self.set_llm_thinking_mode(str(payload.get("thinking_mode") or "standard"))

    def reset(self) -> None:
        defaults = self.defaults()
        self.set_llm_provider(str(defaults["provider"]))
        for provider, model in dict(defaults["models"]).items():
            self.set_llm_model(str(model), str(provider))
        self.set_llm_thinking_mode(str(defaults["thinking_mode"]))
