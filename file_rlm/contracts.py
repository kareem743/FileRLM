from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class QuestionRequest:
    """Input contract for the future RLM engine."""

    file_path: Path
    question: str
    max_root_iterations: int = 26
    max_recursion_depth: int = 1


@dataclass(slots=True)
class AnswerResult:
    """Output contract for the future RLM engine."""

    answer: str
    iterations: int
    subcall_count: int
    recursion_depth: int
    prompt_chars: int
    trace: list[str] = field(default_factory=list)


@dataclass(slots=True)
class REPLExecutionResult:
    """Observable output from one REPL execution step."""

    stdout: str
    state_keys: tuple[str, ...]
    error: str | None = None


class LLMClient(Protocol):
    """Minimal model adapter expected by the engine."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.0,
        timeout_seconds: int = 120,
    ) -> str:
        ...


class REPLRuntime(Protocol):
    """Checkpointed execution environment for the future engine."""

    def initialize(self, *, context: str, question: str, metadata: dict[str, object]) -> None:
        ...

    def execute(self, code: str) -> REPLExecutionResult:
        ...

    def get_variable(self, name: str) -> object:
        ...

    def set_variable(self, name: str, value: object) -> None:
        ...

    def list_variables(self) -> tuple[str, ...]:
        ...

    def close(self) -> None:
        ...


class RecursiveLanguageModel(Protocol):
    """High-level engine contract exposed to the GUI."""

    def answer(self, request: QuestionRequest) -> AnswerResult:
        ...
