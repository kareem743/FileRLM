from pathlib import Path

from file_rlm.config import ModelSettings, RuntimeLimits
from file_rlm.contracts import QuestionRequest
from file_rlm.engine import RLMEngine
from file_rlm.loaders import LoadedDocument
from file_rlm.repl_runtime import REPLExecutionResult


class FakeLoader:
    def load(self, path: Path) -> LoadedDocument:
        return LoadedDocument(
            path=path,
            text="alpha needle omega",
            file_type="txt",
            char_count=18,
            line_count=1,
        )


class FakeOllamaClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *, system_prompt: str, user_prompt: str, model: str) -> str:
        self.calls += 1

        if self.calls == 1:
            return "```repl\nscratch = 'needle found in alpha needle omega'\nprint(scratch)\n```"

        return "FINAL_VAR(scratch)"


class FakeRuntime:
    def __init__(self) -> None:
        self.state: dict[str, object] = {}

    def initialize(self, *, context: str, question: str, metadata: dict[str, object]) -> None:
        self.state["context"] = context
        self.state["query"] = question
        self.state["metadata"] = metadata
        self.state["__subcall_count"] = 0

    def execute(self, code: str) -> REPLExecutionResult:
        self.state["scratch"] = "needle found in alpha needle omega"
        return REPLExecutionResult(
            stdout="needle found in alpha needle omega",
            state_keys=tuple(sorted(self.state)),
        )

    def get_variable(self, name: str) -> object:
        return self.state[name]

    def close(self) -> None:
        return None


def test_engine_can_finish_with_final_var() -> None:
    runtime = FakeRuntime()
    engine = RLMEngine(
        document_loader=FakeLoader(),
        ollama_client=FakeOllamaClient(),
        runtime_factory=lambda: runtime,
        model_settings=ModelSettings(),
        limits=RuntimeLimits(),
    )

    result = engine.answer(
        QuestionRequest(file_path=Path("sample.txt"), question="Where is the needle?")
    )

    assert result.answer == "needle found in alpha needle omega"
    assert result.iterations == 2
    assert result.subcall_count == 0
    assert result.prompt_chars == 18
    assert runtime.state["query"] == "Where is the needle?"


def test_engine_can_finish_with_direct_final() -> None:
    class DirectFinalClient:
        def generate(self, *, system_prompt: str, user_prompt: str, model: str) -> str:
            return "FINAL(The answer is 42.)"

    runtime = FakeRuntime()
    engine = RLMEngine(
        document_loader=FakeLoader(),
        ollama_client=DirectFinalClient(),
        runtime_factory=lambda: runtime,
        model_settings=ModelSettings(),
        limits=RuntimeLimits(),
    )

    result = engine.answer(QuestionRequest(file_path=Path("sample.txt"), question="What is it?"))

    assert result.answer == "The answer is 42."
    assert result.iterations == 1


def test_engine_reprompts_after_invalid_model_output() -> None:
    class InvalidThenFinalClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, *, system_prompt: str, user_prompt: str, model: str) -> str:
            self.prompts.append(user_prompt)
            if len(self.prompts) == 1:
                return "I think the answer is probably in the text."
            return "FINAL(recovered)"

    client = InvalidThenFinalClient()
    engine = RLMEngine(
        document_loader=FakeLoader(),
        ollama_client=client,
        runtime_factory=FakeRuntime,
        model_settings=ModelSettings(),
        limits=RuntimeLimits(),
    )

    result = engine.answer(QuestionRequest(file_path=Path("sample.txt"), question="Recover?"))

    assert result.answer == "recovered"
    assert len(client.prompts) == 2
    assert "invalid" in client.prompts[1].lower()


def test_engine_raises_after_iteration_limit() -> None:
    class AlwaysInvalidClient:
        def generate(self, *, system_prompt: str, user_prompt: str, model: str) -> str:
            return "still invalid"

    engine = RLMEngine(
        document_loader=FakeLoader(),
        ollama_client=AlwaysInvalidClient(),
        runtime_factory=FakeRuntime,
        model_settings=ModelSettings(),
        limits=RuntimeLimits(),
    )

    request = QuestionRequest(
        file_path=Path("sample.txt"),
        question="Will this fail?",
        max_root_iterations=2,
    )

    import pytest

    with pytest.raises(RuntimeError, match="iteration limit"):
        engine.answer(request)


def test_engine_emits_progress_updates() -> None:
    updates: list[str] = []
    runtime = FakeRuntime()
    engine = RLMEngine(
        document_loader=FakeLoader(),
        ollama_client=FakeOllamaClient(),
        runtime_factory=lambda: runtime,
        model_settings=ModelSettings(),
        limits=RuntimeLimits(),
    )

    result = engine.answer(
        QuestionRequest(file_path=Path("sample.txt"), question="Where is the needle?"),
        progress_callback=updates.append,
    )

    assert result.answer == "needle found in alpha needle omega"
    assert any("Depth 0 step 1/" in update for update in updates)
    assert any("Generated REPL code" in update for update in updates)
    assert any("scratch = 'needle found in alpha needle omega'" in update for update in updates)
    assert any("Final answer" in update for update in updates)
