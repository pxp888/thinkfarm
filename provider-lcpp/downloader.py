"""Resumable model downloader with progress tracking and integrity checks."""

import hashlib
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import httpx
except ImportError:
    httpx = None


class DownloadError(Exception):
    """Base exception for download errors."""


class DiskSpaceError(DownloadError):
    """Not enough free disk space."""


class DownloadCancelled(DownloadError):
    """Download was cancelled by user."""


class ChecksumMismatchError(DownloadError):
    """Downloaded file checksum did not match expected."""


@dataclass(frozen=True)
class ModelFileArtifact:
    rel_path: Path
    url: str
    size_bytes: int
    sha256: str | None = None


@dataclass
class ProgressInfo:
    file_index: int              # 1-based
    total_files: int
    filename: str
    file_bytes_done: int
    file_bytes_total: int
    overall_bytes_done: int
    overall_bytes_total: int
    speed_bps: float
    eta_seconds: float | None

    @property
    def file_percent(self) -> float:
        if self.file_bytes_total <= 0:
            return 0.0
        return min(100.0, (self.file_bytes_done / self.file_bytes_total) * 100.0)

    @property
    def overall_percent(self) -> float:
        if self.overall_bytes_total <= 0:
            return 0.0
        return min(100.0, (self.overall_bytes_done / self.overall_bytes_total) * 100.0)


def format_bytes(num_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0 or unit == "TB":
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{int(num_bytes)} B"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} GB"


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds > 86400 * 7:
        return "--:--"
    secs = int(seconds)
    hours, rem = divmod(secs, 3600)
    mins, s = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{s:02d}"
    return f"{mins:02d}:{s:02d}"


def check_disk_space(target_dir: Path, required_bytes: int, safety_margin_gb: float = 1.0) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target_dir)
    margin_bytes = int(safety_margin_gb * (1024 ** 3))
    if usage.free < (required_bytes + margin_bytes):
        needed_fmt = format_bytes(required_bytes + margin_bytes)
        avail_fmt = format_bytes(usage.free)
        raise DiskSpaceError(
            f"Insufficient disk space in {target_dir}. "
            f"Available: {avail_fmt}, Needed (including {safety_margin_gb:.1f}GB safety margin): {needed_fmt}"
        )


def verify_sha256(file_path: Path, expected_hash: str, cancel_event: threading.Event | None = None) -> bool:
    hasher = hashlib.sha256()
    chunk_size = 4 * 1024 * 1024
    with open(file_path, "rb") as f:
        while True:
            if cancel_event and cancel_event.is_set():
                raise DownloadCancelled("Verification cancelled")
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest().lower() == expected_hash.lower()


def download_artifacts(
    artifacts: list[ModelFileArtifact],
    base_dir: Path,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[ProgressInfo], None] | None = None,
    check_hash: bool = True,
) -> None:
    if httpx is None:
        raise DownloadError("httpx library is required for downloading models. Run ./thinkfarm.sh to install it.")

    base_dir.mkdir(parents=True, exist_ok=True)

    # Pre-calculate overall bytes needed
    overall_total = sum(art.size_bytes for art in artifacts)
    
    # Calculate bytes currently on disk (completed files + partial files)
    bytes_on_disk = 0
    bytes_to_fetch = 0
    for art in artifacts:
        target = base_dir / art.rel_path
        part = target.with_name(target.name + ".part")
        if target.exists() and target.stat().st_size == art.size_bytes:
            bytes_on_disk += art.size_bytes
        elif part.exists():
            current_part = min(part.stat().st_size, art.size_bytes)
            bytes_on_disk += current_part
            bytes_to_fetch += (art.size_bytes - current_part)
        else:
            bytes_to_fetch += art.size_bytes

    if bytes_to_fetch > 0:
        check_disk_space(base_dir, bytes_to_fetch)

    overall_done = bytes_on_disk
    total_files = len(artifacts)

    # Network client with redirects and generous timeouts
    transport = httpx.HTTPTransport(retries=3)
    timeout = httpx.Timeout(connect=20.0, read=60.0, write=20.0, pool=20.0)

    with httpx.Client(transport=transport, timeout=timeout, follow_redirects=True) as client:
        for idx, art in enumerate(artifacts, 1):
            if cancel_event and cancel_event.is_set():
                raise DownloadCancelled("Download cancelled by user")

            target_path = base_dir / art.rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            part_path = target_path.with_name(target_path.name + ".part")

            # 1. Already complete?
            if target_path.exists():
                cur_size = target_path.stat().st_size
                if cur_size == art.size_bytes:
                    continue
                # Corrupted or outdated size — remove and re-download
                target_path.unlink()

            # 2. Check resume offset on .part file
            resume_offset = 0
            if part_path.exists():
                cur_part = part_path.stat().st_size
                if cur_part > art.size_bytes:
                    part_path.unlink()
                else:
                    resume_offset = cur_part

            file_done = resume_offset
            headers = {}
            if resume_offset > 0:
                headers["Range"] = f"bytes={resume_offset}-"

            # Speed tracking variables (EMA)
            last_time = time.time()
            speed_ema = 0.0
            last_cb_time = 0.0

            def emit_progress():
                nonlocal last_cb_time
                if not progress_callback:
                    return
                now = time.time()
                if now - last_cb_time < 0.1 and file_done < art.size_bytes:
                    return
                last_cb_time = now
                remaining = max(0, art.size_bytes - file_done)
                eta = (remaining / speed_ema) if speed_ema > 1000 else None
                progress_callback(
                    ProgressInfo(
                        file_index=idx,
                        total_files=total_files,
                        filename=art.rel_path.name,
                        file_bytes_done=file_done,
                        file_bytes_total=art.size_bytes,
                        overall_bytes_done=overall_done,
                        overall_bytes_total=overall_total,
                        speed_bps=speed_ema,
                        eta_seconds=eta,
                    )
                )

            # Initial progress event for this file
            emit_progress()

            with client.stream("GET", art.url, headers=headers) as response:
                if response.status_code == 416:
                    # Range not satisfiable — could mean already complete or invalid range
                    if resume_offset == art.size_bytes:
                        pass
                    else:
                        part_path.unlink(missing_ok=True)
                        resume_offset = 0
                        file_done = 0
                elif response.status_code not in (200, 206):
                    raise DownloadError(f"HTTP error {response.status_code} fetching {art.url}: {response.reason_phrase}")

                file_mode = "ab" if (response.status_code == 206 and resume_offset > 0) else "wb"
                if file_mode == "wb" and resume_offset > 0:
                    # Server did not honor Range, restart from 0
                    overall_done -= resume_offset
                    file_done = 0
                    resume_offset = 0

                chunk_size = 1024 * 1024  # 1 MB chunk
                with open(part_path, file_mode) as out_f:
                    for chunk in response.iter_bytes(chunk_size=chunk_size):
                        if cancel_event and cancel_event.is_set():
                            raise DownloadCancelled("Download cancelled by user")

                        out_f.write(chunk)
                        chunk_len = len(chunk)
                        file_done += chunk_len
                        overall_done += chunk_len

                        now = time.time()
                        dt = now - last_time
                        if dt > 0:
                            instant_speed = chunk_len / dt
                            alpha = 0.2
                            speed_ema = (alpha * instant_speed + (1 - alpha) * speed_ema) if speed_ema > 0 else instant_speed
                            last_time = now

                        emit_progress()

            # Final progress emit for completed file
            emit_progress()

            # Verification
            final_part_size = part_path.stat().st_size
            if final_part_size != art.size_bytes:
                raise DownloadError(
                    f"Download size mismatch for {art.rel_path.name}: "
                    f"got {final_part_size} bytes, expected {art.size_bytes} bytes"
                )

            if check_hash and art.sha256:
                if not verify_sha256(part_path, art.sha256, cancel_event):
                    part_path.unlink(missing_ok=True)
                    raise ChecksumMismatchError(f"SHA-256 hash check failed for {art.rel_path.name}")

            # Atomic rename from .part to final
            os.replace(part_path, target_path)
