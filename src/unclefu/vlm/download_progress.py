"""First-launch model-download progress reporting.

Both the VLM worker (mlx-vlm + huggingface_hub) and the TTS worker
(mlx-audio + huggingface_hub) call `snapshot_download` at startup.
First launch this means ~5.2 GB (Gemma) + ~2 GB (Qwen) of cold download
over the network. Without UX feedback the user sees `⏳ Setting up…`
for 5-15 minutes with no signal anything is happening.

Approach: filesystem-based, not tqdm-based. We don't hook into
huggingface_hub's internal progress (fragile across HF Hub versions).
Instead each worker registers its target cache directory + known total
bytes, and the menu bar's refresh tick walks the directory's current
size to compute progress. Cheap (~5 ms for a 7 GB dir), accurate
within rounding, library-agnostic.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class _Item:
    label: str            # display name, e.g. "Gemma 4 VLM"
    cache_dir: Path       # where HF Hub is downloading into
    target_bytes: int     # known total — hardcoded per model id
    started: bool = False
    done: bool = False


@dataclass
class DownloadProgress:
    """Singleton-ish progress tracker. Created at module import time;
    workers register themselves before kicking off snapshot_download."""

    _items: dict[str, _Item] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def register(self, key: str, label: str,
                 cache_dir: Path, target_bytes: int) -> None:
        """Tell the tracker about a model we're about to download."""
        with self._lock:
            self._items[key] = _Item(
                label=label, cache_dir=cache_dir,
                target_bytes=target_bytes,
            )

    def mark_started(self, key: str) -> None:
        """Begin tracking. After this call, status() reports `downloading`
        for the active item until mark_done."""
        with self._lock:
            if key in self._items:
                self._items[key].started = True

    def mark_done(self, key: str) -> None:
        """The download (or cached check) completed. Subsequent status
        snapshots no longer count this item as in-flight."""
        with self._lock:
            if key in self._items:
                self._items[key].done = True

    def status(self) -> dict:
        """Aggregate for menu bar display. Returns one of:
        - {"phase": "idle"}         — nothing registered yet
        - {"phase": "done"}         — every registered item is done
        - {"phase": "downloading", "pct": int, "current_gb": float,
           "target_gb": float, "active_label": str | None}
        """
        with self._lock:
            if not self._items:
                return {"phase": "idle"}
            if all(i.done for i in self._items.values()):
                return {"phase": "done"}

            total_target = sum(i.target_bytes for i in self._items.values())
            current = 0
            for item in self._items.values():
                if item.done:
                    current += item.target_bytes
                elif item.started:
                    current += _dir_size_bytes(item.cache_dir)

            active_label = next(
                (i.label for i in self._items.values() if not i.done),
                None,
            )
            pct = min(99, int(100 * current / total_target)) if total_target else 0
            return {
                "phase": "downloading",
                "pct": pct,
                "current_gb": current / (1024 ** 3),
                "target_gb": total_target / (1024 ** 3),
                "active_label": active_label,
            }


def _dir_size_bytes(path: Path) -> int:
    """Recursive sum of file sizes under `path`. Returns 0 on any
    error (e.g. dir doesn't exist yet because download hasn't started
    writing). Excludes symlinks to avoid double-counting HF Hub's
    blobs↔snapshots indirection."""
    if not path.exists():
        return 0
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file() and not p.is_symlink():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


# Module-level singleton — both workers and the menu bar share this.
PROGRESS = DownloadProgress()


# Known model sizes — hardcoded because we register before download
# starts (no network call needed). Update if we change models.
KNOWN_MODEL_SIZES_BYTES: dict[str, int] = {
    # mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit ≈ 2.0 GB
    "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit": 2_100_000_000,
    # mlx-community/gemma-4-e4b-it-4bit ≈ 5.2 GB
    "mlx-community/gemma-4-e4b-it-4bit": 5_500_000_000,
}


def hf_cache_path_for(repo_id: str) -> Path:
    """Compute the HF Hub cache directory for a repo id without
    importing huggingface_hub (avoid import-time overhead)."""
    import os
    base = Path(os.environ.get(
        "HF_HOME", Path.home() / ".cache" / "huggingface"
    ))
    flat = repo_id.replace("/", "--")
    return base / "hub" / f"models--{flat}"
