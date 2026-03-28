from __future__ import annotations

from pathlib import Path
from typing import Callable


Downloader = Callable[..., str]


def _hf_download(*, repo_id: str, filename: str, local_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
    )


def ensure_model_file(
    *,
    repo_id: str,
    filename: str,
    models_dir: Path,
    downloader: Downloader | None = None,
) -> Path:
    """Resolve a GGUF file under the project model store and download only if missing."""

    models_dir.mkdir(parents=True, exist_ok=True)
    target_path = models_dir / filename
    if target_path.exists():
        return target_path

    download_fn = downloader or _hf_download
    downloaded = download_fn(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(models_dir),
    )
    downloaded_path = Path(downloaded)
    if downloaded_path.exists():
        return downloaded_path
    if target_path.exists():
        return target_path

    raise FileNotFoundError(
        f"Model download did not produce `{filename}` in `{models_dir}`."
    )
