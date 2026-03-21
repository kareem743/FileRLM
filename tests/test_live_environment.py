import subprocess
from pathlib import Path

import pytest

from file_rlm.config import DockerSettings, ModelSettings, RuntimeLimits
from file_rlm.contracts import QuestionRequest
from file_rlm.engine import RLMEngine
from file_rlm.loaders import DocumentLoader
from file_rlm.ollama_client import OllamaHTTPClient
from file_rlm.repl_runtime import DockerREPLRuntime


@pytest.mark.integration
def test_docker_daemon_is_available() -> None:
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.integration
def test_ollama_has_qwen3_8b_installed() -> None:
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "qwen3:8b" in result.stdout


@pytest.mark.integration
def test_docker_repl_runtime_can_execute_python(tmp_path) -> None:
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
def test_ollama_client_can_get_a_response() -> None:
    client = OllamaHTTPClient(host=ModelSettings().ollama_host)
    response = client.generate(
        system_prompt="Reply with exactly one word: OK",
        user_prompt="Say OK",
        model="qwen3:8b",
    )

    assert response.strip()


@pytest.mark.integration
def test_live_engine_can_answer_a_simple_txt_question(tmp_path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "The access code is 314159.\nThis is a smoke test file.",
        encoding="utf-8",
    )

    model_settings = ModelSettings()
    limits = RuntimeLimits(max_root_iterations=4)
    engine = RLMEngine(
        document_loader=DocumentLoader(),
        ollama_client=OllamaHTTPClient(host=model_settings.ollama_host),
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
