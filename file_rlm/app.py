from __future__ import annotations

import sys

from file_rlm.config import DockerSettings, ModelSettings, RuntimeLimits
from file_rlm.engine import RLMEngine
from file_rlm.gui.main_window import MainWindow
from file_rlm.loaders import DocumentLoader
from file_rlm.ollama_client import OllamaHTTPClient
from file_rlm.repl_runtime import DockerREPLRuntime


def build_engine() -> RLMEngine:
    model_settings = ModelSettings()
    limits = RuntimeLimits()
    docker = DockerSettings()
    ollama_client = OllamaHTTPClient(host=model_settings.ollama_host)

    return RLMEngine(
        document_loader=DocumentLoader(),
        ollama_client=ollama_client,
        runtime_factory=lambda: DockerREPLRuntime(
            docker=docker,
            model_settings=model_settings,
            limits=limits,
        ),
        model_settings=model_settings,
        limits=limits,
    )


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = MainWindow(engine=build_engine())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
