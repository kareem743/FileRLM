from __future__ import annotations

from pathlib import Path
from typing import Any

from file_rlm.config import ModelSettings
from file_rlm.model_store import ensure_model_file


class LlamaCppClient:
    """Small adapter that serves root calls through llama.cpp Python bindings."""

    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings
        self._root_llm: Any | None = None

    def _load_root_model(self) -> Any:
        if self._root_llm is not None:
            return self._root_llm

        from llama_cpp import Llama

        root_path = ensure_model_file(
            repo_id=self.settings.root_repo_id,
            filename=self.settings.root_filename,
            models_dir=self.settings.models_dir,
        )
        llama_kwargs: dict[str, object] = {
            "model_path": str(root_path),
            "n_ctx": self.settings.n_ctx,
        }
        if self.settings.n_threads > 0:
            llama_kwargs["n_threads"] = self.settings.n_threads

        self._root_llm = Llama(**llama_kwargs)
        return self._root_llm

    def prepare_subcall_model(self) -> Any:
        """Optional helper for pre-loading subcall model from HF."""

        from llama_cpp import Llama

        self.settings.models_dir.mkdir(parents=True, exist_ok=True)
        return Llama.from_pretrained(
            repo_id=self.settings.subcall_repo_id,
            filename=self.settings.subcall_filename,
            local_dir=str(self.settings.models_dir),
        )

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.0,
        timeout_seconds: int = 120,
    ) -> str:
        del model, timeout_seconds
        llm = self._load_root_model()

        payload = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        choices = payload.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return str(message.get("content", "")).strip()


def resolve_root_model_path(settings: ModelSettings) -> Path:
    return ensure_model_file(
        repo_id=settings.root_repo_id,
        filename=settings.root_filename,
        models_dir=settings.models_dir,
    )
