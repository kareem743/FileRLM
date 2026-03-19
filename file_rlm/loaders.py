from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class LoadedDocument:
    path: Path
    text: str
    file_type: str
    char_count: int
    line_count: int


def _get_pdf_reader_class():
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires the 'pypdf' package.") from exc

    return PdfReader


class DocumentLoader:
    """Loads `.txt` and `.pdf` files into plain text for the RLM runtime."""

    def load(self, path: Path) -> LoadedDocument:
        suffix = path.suffix.lower()

        if suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="replace")
            return LoadedDocument(
                path=path,
                text=text,
                file_type="txt",
                char_count=len(text),
                line_count=text.count("\n") + (1 if text else 0),
            )

        if suffix == ".pdf":
            reader_cls = _get_pdf_reader_class()
            reader = reader_cls(str(path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
            return LoadedDocument(
                path=path,
                text=text,
                file_type="pdf",
                char_count=len(text),
                line_count=text.count("\n") + (1 if text else 0),
            )

        raise ValueError(f"Unsupported file type: {path.suffix}")
