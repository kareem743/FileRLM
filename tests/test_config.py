from file_rlm.config import AppConfig, DockerSettings, ModelSettings, RuntimeLimits


def test_model_settings_are_local_llama_cpp_friendly() -> None:
    settings = ModelSettings()

    assert settings.root_repo_id == "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"
    assert settings.root_filename == "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
    assert settings.subcall_model == "Qwen_Qwen3-8B-Q4_K_M.gguf"
    assert settings.subcall_repo_id == "bartowski/Qwen_Qwen3-8B-GGUF"
    assert settings.subcall_filename == "Qwen_Qwen3-8B-Q4_K_M.gguf"
    assert settings.models_dir.name == "gguf"
    assert settings.models_dir.parent.name == "models"
    assert settings.temperature == 0.0


def test_runtime_limits_are_positive() -> None:
    limits = RuntimeLimits()

    assert limits.max_root_iterations > 0
    assert limits.max_recursion_depth >= 0
    assert limits.max_chars_per_subquery > 0
    assert limits.max_stdout_chars > 0


def test_docker_settings_target_model_gateway_from_container() -> None:
    docker = DockerSettings()

    assert docker.image == "python:3.11-slim"
    assert docker.model_gateway_url == "http://host.docker.internal:8010/v1/subcall"
    assert docker.model_gateway_bind_host == "0.0.0.0"
    assert docker.workdir == "/workspace"


def test_app_config_points_to_simple_repo_directories() -> None:
    config = AppConfig()

    assert config.traces_dir.name == "traces"
    assert config.evals_dir.name == "evals"
    assert config.default_file_encoding == "utf-8"
    assert config.supported_extensions == (".txt", ".pdf")
