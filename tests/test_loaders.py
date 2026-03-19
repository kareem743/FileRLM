from pathlib import Path

import pytest

from file_rlm.loaders import DocumentLoader


def test_text_loader_reads_txt_files(tmp_path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("hello\nworld", encoding="utf-8")

    document = DocumentLoader().load(path)

    assert document.file_type == "txt"
    assert document.text == "hello\nworld"
    assert document.char_count == len("hello\nworld")
    assert document.line_count == 2


def test_pdf_loader_uses_pdf_reader(tmp_path, monkeypatch) -> None:
    class FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class FakeReader:
        def __init__(self, _: str) -> None:
            self.pages = [FakePage("first page"), FakePage("second page")]

    monkeypatch.setattr("file_rlm.loaders._get_pdf_reader_class", lambda: FakeReader)

    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF-1.4")

    document = DocumentLoader().load(path)

    assert document.file_type == "pdf"
    assert "first page" in document.text
    assert "second page" in document.text


def test_loader_rejects_unsupported_extensions(tmp_path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# unsupported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        DocumentLoader().load(path)
