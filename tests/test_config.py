from file_rlm.config import AppConfig, DockerSettings, ModelSettings, RuntimeLimits


def test_model_settings_are_local_ollama_friendly() -> None:
    settings = ModelSettings()

    assert settings.root_model == "qwen3:8b"
    assert settings.subcall_model == "qwen3:8b"
    assert settings.ollama_host.startswith("http://")
    assert settings.temperature == 0.0


def test_runtime_limits_are_positive() -> None:
    limits = RuntimeLimits()

    assert limits.max_root_iterations > 0
    assert limits.max_recursion_depth >= 0
    assert limits.max_chars_per_subquery > 0
    assert limits.max_stdout_chars > 0


def test_docker_settings_target_host_ollama_from_container() -> None:
    docker = DockerSettings()

    assert docker.image == "python:3.11-slim"
    assert docker.ollama_url == "http://host.docker.internal:11434"
    assert docker.workdir == "/workspace"


def test_app_config_points_to_simple_repo_directories() -> None:
    config = AppConfig()

    assert config.traces_dir.name == "traces"
    assert config.evals_dir.name == "evals"
    assert config.default_file_encoding == "utf-8"
    assert config.supported_extensions == (".txt", ".pdf")
