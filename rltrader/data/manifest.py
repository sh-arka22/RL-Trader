"""Build manifest: checksums, row counts, gate results.

``make data`` twice must produce identical file checksums (PLAN.md S2 exit criterion).
That only holds if nothing volatile leaks into the Parquet payload, so ``recorded_at``
and ``ingest_id`` are pinned by the caller when reproducibility is being tested.
"""
from __future__ import annotations
import json
import pathlib
import platform
import sys
import datetime as dt
from .store import checksum


def build_manifest(root: str | pathlib.Path, extra: dict | None = None) -> dict:
    root = pathlib.Path(root)
    files = sorted(p for p in root.glob("**/*.parquet"))
    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "n_files": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "files": {str(p.relative_to(root)): {"sha256": checksum(p), "bytes": p.stat().st_size}
                  for p in files},
        **(extra or {}),
    }


def write_manifest(root: str | pathlib.Path, path: str | pathlib.Path, extra: dict | None = None) -> dict:
    m = build_manifest(root, extra)
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(m, indent=2, default=str))
    return m


def payload_digest(root: str | pathlib.Path) -> str:
    """Checksum of file CONTENT only, order-stable — the reproducibility fingerprint."""
    import hashlib
    root = pathlib.Path(root)
    h = hashlib.sha256()
    for p in sorted(root.glob("**/*.parquet")):
        h.update(str(p.relative_to(root)).encode())
        h.update(checksum(p).encode())
    return h.hexdigest()
