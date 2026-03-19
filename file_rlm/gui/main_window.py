from __future__ import annotations

from pathlib import Path

try:
    from PyQt6.QtCore import QObject, QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    raise RuntimeError("PyQt6 is required to run the desktop UI.") from exc

from file_rlm.contracts import QuestionRequest


class EngineWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, engine, file_path: str, question: str) -> None:
        super().__init__()
        self.engine = engine
        self.file_path = file_path
        self.question = question

    def run(self) -> None:
        try:
            result = self.engine.answer(
                QuestionRequest(file_path=Path(self.file_path), question=self.question),
                progress_callback=self.progress.emit,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        self.finished.emit(result.answer)


class MainWindow(QMainWindow):
    def __init__(self, engine) -> None:
        super().__init__()
        self.engine = engine
        self.worker_thread: QThread | None = None
        self.worker: EngineWorker | None = None

        self.setWindowTitle("File RLM")
        self.resize(840, 560)

        root = QWidget(self)
        layout = QVBoxLayout(root)

        path_row = QHBoxLayout()
        path_label = QLabel("File")
        self.path_input = QLineEdit()
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._browse_for_file)
        path_row.addWidget(path_label)
        path_row.addWidget(self.path_input)
        path_row.addWidget(browse_button)

        layout.addLayout(path_row)

        self.question_input = QTextEdit()
        self.question_input.setPlaceholderText("Ask a question about the selected PDF or TXT file.")
        layout.addWidget(self.question_input)

        ask_button = QPushButton("Ask")
        ask_button.clicked.connect(self._ask_question)
        layout.addWidget(ask_button)

        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        layout.addWidget(self.answer_output)

        self.setCentralWidget(root)

    def _browse_for_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a file",
            "",
            "Documents (*.txt *.pdf)",
        )
        if file_path:
            self.path_input.setText(file_path)

    def _ask_question(self) -> None:
        file_path = self.path_input.text().strip()
        question = self.question_input.toPlainText().strip()
        if not file_path or not question:
            QMessageBox.warning(self, "Missing input", "Choose a file and enter a question.")
            return

        self.answer_output.setPlainText("Running...")
        self.worker_thread = QThread(self)
        self.worker = EngineWorker(self.engine, file_path, question)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_answer_ready)
        self.worker.failed.connect(self._on_answer_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.start()

    def _on_progress(self, message: str) -> None:
        current = self.answer_output.toPlainText().strip()
        if current == "Running...":
            self.answer_output.setPlainText(message)
            return
        if current:
            self.answer_output.append("")
        self.answer_output.append(message)

    def _on_answer_ready(self, answer: str) -> None:
        current = self.answer_output.toPlainText().strip()
        if current:
            self.answer_output.append("")
            self.answer_output.append(f"Answer:\n{answer}")
            return
        self.answer_output.setPlainText(answer)

    def _on_answer_failed(self, error: str) -> None:
        self.answer_output.setPlainText(f"Error: {error}")
