from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ModelSettings:
    """Configures local llama.cpp models used by the RLM engine."""

    root_model: str = "qwen2.5-coder-7b-instruct-q4_k_m"
    subcall_model: str = "qwen3-8b-gguf"
    root_repo_id: str = "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"
    root_filename: str = "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
    subcall_repo_id: str = "bartowski/Qwen_Qwen3-8B-GGUF"
    subcall_filename: str = "GGUF_FILE"
    models_dir: Path = field(default_factory=lambda: Path.cwd() / "models" / "gguf")
    temperature: float = 0.0
    request_timeout_seconds: int = 120
    n_ctx: int = 8_192
    n_threads: int = 0


@dataclass(slots=True)
class RuntimeLimits:
    """Hard limits for the planning-stage RLM contract."""

    max_root_iterations: int = 20
    max_recursion_depth: int = 3
    max_recursive_calls: int = 8
    max_chars_per_subquery: int = 24_000
    max_stdout_chars: int = 4_000
    max_subcalls: int = 32
    max_repl_code_chars: int = 20_000
    repl_timeout_seconds: int = 30
    max_observation_chars: int = 6_000
    max_child_iterations: int = 5


@dataclass(slots=True)
class DockerSettings:
    """Docker settings for the isolated REPL runtime."""

    image: str = "python:3.11-slim"
    model_gateway_url: str = "http://host.docker.internal:8010"
    workdir: str = "/workspace"
    memory_limit: str = "512m"
    cpu_limit: str = "1.0"
    pids_limit: int = 64
    user: str = "65534:65534"


@dataclass(slots=True)
class AppConfig:
    """Small app-level defaults used across tests and future implementation."""

    workspace_dir: Path = field(default_factory=Path.cwd)
    traces_dir: Path = field(default_factory=lambda: Path.cwd() / "traces")
    evals_dir: Path = field(default_factory=lambda: Path.cwd() / "evals")
    default_file_encoding: str = "utf-8"
    supported_extensions: tuple[str, ...] = (".txt", ".pdf")
