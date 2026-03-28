import importlib
import subprocess
from pathlib import Path

import pytest

from file_rlm.config import DockerSettings, ModelSettings, RuntimeLimits
from file_rlm.contracts import QuestionRequest
from file_rlm.engine import RLMEngine
from file_rlm.llama_cpp_client import LlamaCppClient
from file_rlm.loaders import DocumentLoader
from file_rlm.repl_runtime import DockerREPLRuntime


def require_docker_daemon() -> None:
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(result.stderr or result.stdout or "Docker daemon is unavailable.")


def require_llama_cpp_runtime() -> None:
    try:
        importlib.import_module("llama_cpp")
    except Exception as exc:  # pragma: no cover - dependent on local machine state.
        pytest.skip(f"llama_cpp runtime is unavailable: {exc}")


@pytest.mark.integration
def test_docker_daemon_is_available() -> None:
    require_docker_daemon()


@pytest.mark.integration
def test_llama_cpp_dependency_is_installed() -> None:
    require_llama_cpp_runtime()


@pytest.mark.integration
def test_docker_repl_runtime_can_execute_python(tmp_path) -> None:
    require_docker_daemon()

    runtime = DockerREPLRuntime(
        docker=DockerSettings(),
        model_settings=ModelSettings(),
        limits=RuntimeLimits(),
        workspace_root=tmp_path,
    )
    runtime.initialize(
        context="The access code is 314159.",
        question="What is the code?",
        metadata={"path": "sample.txt"},
    )

    result = runtime.execute("answer = context.split()[-1].strip('.'); print(answer)")

    assert "314159" in result.stdout
    assert runtime.get_variable("answer") == "314159"


@pytest.mark.integration
def test_llama_cpp_client_can_get_a_response() -> None:
    require_llama_cpp_runtime()
    settings = ModelSettings()
    model_path = settings.models_dir / settings.root_filename
    if not model_path.exists():
        pytest.skip(f"Local model file is missing: {model_path}")

    client = LlamaCppClient(settings=settings)
    response = client.generate(
        system_prompt="Reply with exactly one word: OK",
        user_prompt="Say OK",
        model=settings.root_model,
    )

    assert response.strip()


@pytest.mark.integration
def test_live_engine_can_answer_a_simple_txt_question(tmp_path) -> None:
    require_docker_daemon()
    require_llama_cpp_runtime()

    model_settings = ModelSettings()
    model_path = model_settings.models_dir / model_settings.root_filename
    if not model_path.exists():
        pytest.skip(f"Local model file is missing: {model_path}")

    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "The access code is 314159.\nThis is a smoke test file.",
        encoding="utf-8",
    )

    limits = RuntimeLimits(max_root_iterations=4)
    engine = RLMEngine(
        document_loader=DocumentLoader(),
        llm_client=LlamaCppClient(settings=model_settings),
        runtime_factory=lambda: DockerREPLRuntime(
            docker=DockerSettings(),
            model_settings=model_settings,
            limits=limits,
        ),
        model_settings=model_settings,
        limits=limits,
    )

    result = engine.answer(
        QuestionRequest(
            file_path=file_path,
            question="What is the access code?",
            max_root_iterations=4,
        )
    )

    assert "314159" in result.answer
