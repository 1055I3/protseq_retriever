"""First-boot data bootstrap for the bioseq_retriever pipeline.

Downloads per-protein.h5 (~1.3 GB) and, optionally, a pre-built FAISS
index + accession cache into BIOSEQ_DATA_DIR before the pipeline tries
to read them. Idempotent: a second call is a no-op if the files are
already on disk.

Source is selected by `BIOSEQ_DATA_SOURCE`:

  "uniprot"               UniProt FTP (slow, no setup required).
  "hf:OWNER/DATASET_REPO" HF Hub dataset (fast, requires one-time upload
                          of per-protein.h5 [+ optional .index/.pkl] to
                          that dataset by the operator).

Default is "uniprot" so that the Space boots without extra setup, even
if cold starts take ~10 min on the embeddings download.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_UNIPROT_H5_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/embeddings/uniprot_sprot/per-protein.h5"
)

# A non-empty .h5 must be at least this big — guards against half-downloaded
# files left behind by a killed container. Anything smaller is treated as
# missing and re-downloaded.
_H5_MIN_BYTES = 1_000_000_000  # 1.0 GB; real file is ~1.38 GB


def _data_dir() -> Path:
    return Path(
        os.getenv("BIOSEQ_DATA_DIR", os.path.join("bioseq_retriever", "data"))
    )


def _file_ok(path: Path, min_bytes: int = 1) -> bool:
    return path.exists() and path.stat().st_size >= min_bytes


def _log(msg: str) -> None:
    print(f"[bioseq.bootstrap] {msg}", flush=True)


def _download_url(url: str, dest: Path) -> None:
    """Stream a URL to disk with progress lines (HF Space logs friendly)."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    _log(f"GET {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bytes_done = 0
        next_log_at = 50 * 1024 * 1024  # log every ~50 MB
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                bytes_done += len(chunk)
                if bytes_done >= next_log_at:
                    pct = (bytes_done / total * 100) if total else 0
                    _log(f"  {bytes_done / 1e9:.2f} GB / {total / 1e9:.2f} GB ({pct:.0f}%)")
                    next_log_at += 50 * 1024 * 1024
    tmp.replace(dest)
    _log(f"saved {dest} ({dest.stat().st_size:,} bytes)")


def _download_hf(repo_id: str, filename: str, dest: Path) -> None:
    """Pull `filename` from a HF Hub dataset repo into `dest`."""
    from huggingface_hub import hf_hub_download  # lazy import

    _log(f"HF Hub: {repo_id}::{filename}")
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(dest.parent),
    )
    cached_path = Path(cached)
    if cached_path != dest:
        if dest.exists():
            dest.unlink()
        cached_path.replace(dest)
    _log(f"saved {dest} ({dest.stat().st_size:,} bytes)")


def ensure_data() -> None:
    """Make sure per-protein.h5 (and optionally .index/.pkl) are on disk.

    Reads BIOSEQ_DATA_SOURCE to pick the source. Safe to call multiple
    times — a second call is a no-op when files are already present.
    """
    data_dir = _data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    h5_path = data_dir / "per-protein.h5"
    index_path = data_dir / "per-protein.index"
    cache_path = data_dir / "per-protein.accessions.pkl"

    source = os.getenv("BIOSEQ_DATA_SOURCE", "uniprot").strip()

    if _file_ok(h5_path, _H5_MIN_BYTES):
        _log(f"per-protein.h5 already present at {h5_path}, skipping download")
    else:
        _log(f"per-protein.h5 missing or undersized at {h5_path}")
        if source.startswith("hf:"):
            repo_id = source.split(":", 1)[1]
            _download_hf(repo_id, "per-protein.h5", h5_path)
        elif source == "uniprot":
            _download_url(_UNIPROT_H5_URL, h5_path)
        else:
            raise RuntimeError(
                f"Unknown BIOSEQ_DATA_SOURCE={source!r}. "
                "Use 'uniprot' or 'hf:OWNER/DATASET'."
            )

    # Pre-built index is optional: only attempt if source is HF (UniProt
    # FTP doesn't host the index — it gets built on first run from the
    # .h5 file by embeddings.get_or_create_index).
    if source.startswith("hf:") and not _file_ok(index_path):
        repo_id = source.split(":", 1)[1]
        try:
            _download_hf(repo_id, "per-protein.index", index_path)
            _download_hf(repo_id, "per-protein.accessions.pkl", cache_path)
            _log("pre-built FAISS index pulled from HF — skipping rebuild")
        except Exception as exc:  # noqa: BLE001
            # Missing index in the dataset is fine — embeddings.py will
            # rebuild from the .h5 (5–15 min one-time cost per cold start).
            _log(f"pre-built index not available ({exc}); will rebuild from .h5")


if __name__ == "__main__":
    ensure_data()
    sys.exit(0)
