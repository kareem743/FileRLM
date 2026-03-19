import json
import pickle
from pathlib import Path

import pytest

from file_rlm.config import DockerSettings, ModelSettings, RuntimeLimits
from file_rlm.repl_runtime import CommandResult, DockerREPLRuntime


def test_docker_runtime_builds_expected_command() -> None:
    runtime = DockerREPLRuntime(
        docker=DockerSettings(image="python:3.11-slim"),
        model_settings=ModelSettings(),
        limits=RuntimeLimits(),
    )

    command = runtime._build_docker_command(Path("C:/tmp/runtime"))

    assert command[:3] == ["docker", "run", "--rm"]
    assert "python:3.11-slim" in command
    assert "C:/tmp/runtime:/workspace" in command
    assert "runner.py" in command[-1]


def test_docker_runtime_updates_state_from_runner(tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_runner(command, workdir):
        captured["command"] = command
        state = pickle.loads((workdir / "state_in.pkl").read_bytes())
        state["buffer"] = state["context"][:5]
        state["__subcall_count"] = 2
        (workdir / "state_out.pkl").write_bytes(pickle.dumps(state))
        (workdir / "result.json").write_text(
            json.dumps(
                {
                    "stdout": "abcde",
                    "state_keys": sorted(state),
                    "error": None,
                }
            ),
            encoding="utf-8",
        )
        return CommandResult(returncode=0, stdout="", stderr="")

    runtime = DockerREPLRuntime(
        docker=DockerSettings(image="python:3.11-slim"),
        model_settings=ModelSettings(),
        limits=RuntimeLimits(),
        runner=fake_runner,
        workspace_root=tmp_path,
    )
    runtime.initialize(context="abcdef", question="What is the prefix?", metadata={"path": "sample.txt"})

    result = runtime.execute("buffer = context[:5]")

    assert captured["command"][0] == "docker"
    assert result.stdout == "abcde"
    assert runtime.get_variable("buffer") == "abcde"
    assert runtime.get_variable("__subcall_count") == 2


def test_docker_runtime_requires_initialize_before_execute(tmp_path) -> None:
    runtime = DockerREPLRuntime(
        docker=DockerSettings(),
        model_settings=ModelSettings(),
        limits=RuntimeLimits(),
        workspace_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="initialized"):
        runtime.execute("print('hello')")


def test_docker_runtime_surfaces_runner_failures(tmp_path) -> None:
    def failing_runner(command, workdir):
        return CommandResult(returncode=1, stdout="", stderr="container failed")

    runtime = DockerREPLRuntime(
        docker=DockerSettings(),
        model_settings=ModelSettings(),
        limits=RuntimeLimits(),
        runner=failing_runner,
        workspace_root=tmp_path,
    )
    runtime.initialize(context="abcdef", question="q", metadata={})

    with pytest.raises(RuntimeError, match="container failed"):
        runtime.execute("print('hello')")


def test_docker_runner_script_explains_disabled_imports(tmp_path) -> None:
    runtime = DockerREPLRuntime(
        docker=DockerSettings(),
        model_settings=ModelSettings(),
        limits=RuntimeLimits(),
        workspace_root=tmp_path,
    )

    workdir = tmp_path / "docker_runtime"
    workdir.mkdir(parents=True, exist_ok=True)
    runtime._write_runner_script(workdir)
    script = (workdir / "runner.py").read_text(encoding="utf-8")

    assert "Imports are disabled in this REPL" in script
    assert "Use the preloaded variables" in script
