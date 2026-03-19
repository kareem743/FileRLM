from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ModelSettings:
    """Configures the local Ollama models used by the future RLM engine."""

    root_model: str = "qwen3-coder:30b"
    subcall_model: str = "qwen3:8b"
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.0


@dataclass(slots=True)
class RuntimeLimits:
    """Hard limits for the planning-stage RLM contract."""

    max_root_iterations: int = 12
    max_recursion_depth: int = 1
    max_chars_per_subquery: int = 24_000
    max_stdout_chars: int = 4_000


@dataclass(slots=True)
class DockerSettings:
    """Docker settings for the isolated REPL runtime."""

    image: str = "python:3.11-slim"
    ollama_url: str = "http://host.docker.internal:11434"
    workdir: str = "/workspace"


@dataclass(slots=True)
class AppConfig:
    """Small app-level defaults used across tests and future implementation."""

    workspace_dir: Path = field(default_factory=Path.cwd)
    traces_dir: Path = field(default_factory=lambda: Path.cwd() / "traces")
    evals_dir: Path = field(default_factory=lambda: Path.cwd() / "evals")
    default_file_encoding: str = "utf-8"
    supported_extensions: tuple[str, ...] = (".txt", ".pdf")
