"""Runtime binary management (llama-server + optional CUDA runtime).

Downloads and caches official ggml-org/llama.cpp release binaries into
~/.thinkfarm/runtime/<os>-<backend>-<version>/ with SHA-256 verification.
Falls back to pre-bundled local bin/ (and cudart/) folders when present so
the existing tarball/zip layout and local development continue to work.
"""

from __future__ import annotations

import glob
import os
import shutil
import stat
import sys
import tarfile
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from downloader import (
    DownloadCancelled,
    DownloadError,
    ModelFileArtifact,
    ProgressInfo,
    download_artifacts,
)

RUNTIME_VERSION = "b12431"
DATA_DIR = Path.home() / ".thinkfarm"
RUNTIMES_DIR = DATA_DIR / "runtime"

BACKEND_LABELS: dict[str, str] = {
    "cuda": "CUDA (NVIDIA)",
    "vulkan": "Vulkan (AMD / Intel / NVIDIA)",
}

_RELEASE_BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{RUNTIME_VERSION}"


@dataclass(frozen=True)
class RuntimeArchive:
    artifact: ModelFileArtifact
    target_subdir: str          # "bin" or "cudart"
    strip_components: int = 0   # tarballs have a single top-level directory


RUNTIME_ARCHIVES: dict[tuple[str, str], tuple[RuntimeArchive, ...]] = {
    ("linux", "cuda"): (
        RuntimeArchive(
            artifact=ModelFileArtifact(
                rel_path=Path(f"llama-{RUNTIME_VERSION}-bin-ubuntu-cuda-13.5-x64.tar.gz"),
                url=f"{_RELEASE_BASE}/llama-{RUNTIME_VERSION}-bin-ubuntu-cuda-13.5-x64.tar.gz",
                size_bytes=202476961,
                sha256="5a3f86d12464f0b44769b7087559930c672b654e318eaf11a08e653175593b73",
            ),
            target_subdir="bin",
            strip_components=1,
        ),
        RuntimeArchive(
            artifact=ModelFileArtifact(
                rel_path=Path(f"cudart-llama-{RUNTIME_VERSION}-bin-ubuntu-cuda-13.5-x64.tar.gz"),
                url=f"{_RELEASE_BASE}/cudart-llama-{RUNTIME_VERSION}-bin-ubuntu-cuda-13.5-x64.tar.gz",
                size_bytes=444077899,
                sha256="59ec2e078d20e5f9ecb06549ec8c2ed91388a27a05f20c8a08528831072f6c15",
            ),
            target_subdir="cudart",
            strip_components=1,
        ),
    ),
    ("linux", "vulkan"): (
        RuntimeArchive(
            artifact=ModelFileArtifact(
                rel_path=Path(f"llama-{RUNTIME_VERSION}-bin-ubuntu-vulkan-x64.tar.gz"),
                url=f"{_RELEASE_BASE}/llama-{RUNTIME_VERSION}-bin-ubuntu-vulkan-x64.tar.gz",
                size_bytes=36387080,
                sha256="135a14e30fd82534f5dd6da3f62e336985102a685006cdde9624de0be8100aa3",
            ),
            target_subdir="bin",
            strip_components=1,
        ),
    ),
    ("win", "cuda"): (
        RuntimeArchive(
            artifact=ModelFileArtifact(
                rel_path=Path(f"llama-{RUNTIME_VERSION}-bin-win-cuda-13.5-x64.zip"),
                url=f"{_RELEASE_BASE}/llama-{RUNTIME_VERSION}-bin-win-cuda-13.5-x64.zip",
                size_bytes=203526771,
                sha256="8ad4fe30e7fd33fc08309975e373b4270498c39d70f17868e100b9bc2736a0d5",
            ),
            target_subdir="bin",
            strip_components=0,
        ),
        RuntimeArchive(
            artifact=ModelFileArtifact(
                rel_path=Path("cudart-llama-bin-win-cuda-13.5-x64.zip"),
                url=f"{_RELEASE_BASE}/cudart-llama-bin-win-cuda-13.5-x64.zip",
                size_bytes=426314199,
                sha256="c2bf4cc202bfbaaa7a0b75f5082457c3811326947bb216059b72839d844e9cba",
            ),
            target_subdir="bin",
            strip_components=0,
        ),
    ),
    ("win", "vulkan"): (
        RuntimeArchive(
            artifact=ModelFileArtifact(
                rel_path=Path(f"llama-{RUNTIME_VERSION}-bin-win-vulkan-x64.zip"),
                url=f"{_RELEASE_BASE}/llama-{RUNTIME_VERSION}-bin-win-vulkan-x64.zip",
                size_bytes=38288583,
                sha256="5b7ccf07dfbb07eb857b165ed2e046971672f300b6ea085b758c534b9acd5d44",
            ),
            target_subdir="bin",
            strip_components=0,
        ),
    ),
}


def current_os_key() -> str:
    return "win" if sys.platform == "win32" else "linux"


def server_binary_name() -> str:
    return "llama-server.exe" if sys.platform == "win32" else "llama-server"


def find_driver_dirs() -> list[str]:
    """Locate the NVIDIA driver's libcuda.so.1 on Linux (including NixOS)."""
    found: list[str] = []
    for pat in (
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/nix/store/*-nvidia-x11-[0-9.]*-lib/lib",
        "/nix/store/*-nvidia-x11-[0-9.]*/lib",
    ):
        for d in glob.glob(pat):
            if Path(d, "libcuda.so.1").exists() and d not in found:
                found.append(d)
    return found


def detect_default_backend() -> str:
    """Auto-detect CUDA if an NVIDIA driver is present, otherwise Vulkan."""
    if sys.platform == "win32":
        sys_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        if (sys_root / "System32" / "nvcuda.dll").exists():
            return "cuda"
        if shutil.which("nvidia-smi"):
            return "cuda"
        return "vulkan"
    if find_driver_dirs() or shutil.which("nvidia-smi"):
        return "cuda"
    return "vulkan"


def get_runtime_dir(backend: str, version: str = RUNTIME_VERSION) -> Path:
    return RUNTIMES_DIR / f"{current_os_key()}-{backend}-{version}"


def _local_bundle_paths(root: Path) -> tuple[Path, Path, Path] | None:
    """Return local pre-bundled (bin_dir, cudart_dir, server_bin) if present."""
    bin_dir = root / "bin" if (root / "bin").exists() else root / "llama-b11064"
    cudart_dir = (
        root / "cudart"
        if (root / "cudart").exists()
        else root / "cudart-llama-b11064-bin-ubuntu-cuda-13.3-x64"
    )
    server_bin = bin_dir / server_binary_name()
    if server_bin.exists():
        return bin_dir, cudart_dir, server_bin
    return None


def _dir_is_cuda(bin_dir: Path, cudart_dir: Path) -> bool:
    if cudart_dir.exists():
        return True
    return (bin_dir / "libggml-cuda.so").exists() or (bin_dir / "ggml-cuda.dll").exists()


def resolve_runtime_paths(backend: str, root: Path) -> tuple[Path, Path, Path, bool]:
    """Resolve (bin_dir, cudart_dir, server_bin, is_local_bundle) for a backend.

    Prefers a local pre-bundled bin/ folder only when it matches the requested
    backend (or if not running as a frozen single-file executable and the local
    bundle is already present).
    """
    local = _local_bundle_paths(root)
    if local is not None:
        l_bin, l_cudart, l_srv = local
        local_backend = "cuda" if _dir_is_cuda(l_bin, l_cudart) else "vulkan"
        if local_backend == backend:
            return l_bin, l_cudart, l_srv, True

    rt_dir = get_runtime_dir(backend)
    bin_dir = rt_dir / "bin"
    cudart_dir = rt_dir / "cudart"
    server_bin = bin_dir / server_binary_name()
    return bin_dir, cudart_dir, server_bin, False


def required_runtime_binaries(backend: str, root: Path) -> list[Path]:
    """Binaries and runtime libraries required to launch llama-server."""
    bin_dir, cudart_dir, server_bin, _ = resolve_runtime_paths(backend, root)
    bins = [server_bin]
    if backend == "cuda":
        if sys.platform == "win32":
            cuda_dlls = list(cudart_dir.glob("cudart64_*.dll")) + list(bin_dir.glob("cudart64_*.dll"))
            bins.append(cuda_dlls[0] if cuda_dlls else bin_dir / "cudart64_13.dll")
        else:
            cudarts = list(cudart_dir.glob("libcudart.so.*")) + list(bin_dir.glob("libcudart.so.*"))
            bins.append(cudarts[0] if cudarts else cudart_dir / "libcudart.so.13")
    return bins


def runtime_status(backend: str, root: Path) -> dict:
    """Return install & download status for the requested backend runtime."""
    os_key = current_os_key()
    archives = RUNTIME_ARCHIVES.get((os_key, backend), ())
    total_bytes = sum(a.artifact.size_bytes for a in archives)

    bin_dir, cudart_dir, server_bin, is_local = resolve_runtime_paths(backend, root)
    req = required_runtime_binaries(backend, root)
    all_bins_present = all(p.exists() for p in req)

    if is_local and all_bins_present:
        return {
            "installed": True,
            "source": "bundled",
            "runtime_dir": bin_dir.parent,
            "total_bytes": total_bytes,
            "downloaded_bytes": total_bytes,
            "remaining_bytes": 0,
        }

    rt_dir = get_runtime_dir(backend)
    ready_marker = rt_dir / ".ready"
    if ready_marker.exists() and all_bins_present:
        return {
            "installed": True,
            "source": "cached",
            "runtime_dir": rt_dir,
            "total_bytes": total_bytes,
            "downloaded_bytes": total_bytes,
            "remaining_bytes": 0,
        }

    # Check partial archive downloads in _archives/
    archive_dir = rt_dir / "_archives"
    downloaded_bytes = 0
    for ra in archives:
        art = ra.artifact
        target = archive_dir / art.rel_path
        part = target.with_name(target.name + ".part")
        if target.exists() and target.stat().st_size == art.size_bytes:
            downloaded_bytes += art.size_bytes
        elif part.exists():
            downloaded_bytes += min(part.stat().st_size, art.size_bytes)

    return {
        "installed": False,
        "source": "missing",
        "runtime_dir": rt_dir,
        "total_bytes": total_bytes,
        "downloaded_bytes": downloaded_bytes,
        "remaining_bytes": max(0, total_bytes - downloaded_bytes),
    }


def _safe_extract_tar(archive_path: Path, dest_dir: Path, strip_components: int = 0) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            parts = Path(member.name).parts
            # Drop leading '.' or '/' components if any
            parts = tuple(p for p in parts if p not in ("", "."))
            if len(parts) <= strip_components:
                continue
            rel_parts = parts[strip_components:]
            if any(p == ".." for p in rel_parts):
                raise DownloadError(f"Unsafe path in archive: {member.name}")
            out_path = (dest_dir / Path(*rel_parts)).resolve()
            if dest_resolved != out_path and dest_resolved not in out_path.parents:
                raise DownloadError(f"Archive entry escapes destination: {member.name}")

            if member.isdir():
                out_path.mkdir(parents=True, exist_ok=True)
            elif member.issym():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists() or out_path.is_symlink():
                    out_path.unlink()
                os.symlink(member.linkname, out_path)
            elif member.isfile():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    continue
                with src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                if member.mode:
                    os.chmod(out_path, stat.S_IMODE(member.mode))


def _safe_extract_zip(archive_path: Path, dest_dir: Path, strip_components: int = 0) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()
    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            parts = tuple(p for p in Path(info.filename).parts if p not in ("", "."))
            if len(parts) <= strip_components:
                continue
            rel_parts = parts[strip_components:]
            if any(p == ".." for p in rel_parts):
                raise DownloadError(f"Unsafe path in archive: {info.filename}")
            out_path = (dest_dir / Path(*rel_parts)).resolve()
            if dest_resolved != out_path and dest_resolved not in out_path.parents:
                raise DownloadError(f"Archive entry escapes destination: {info.filename}")

            if info.is_dir():
                out_path.mkdir(parents=True, exist_ok=True)
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)


def download_and_extract_runtime(
    backend: str,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[ProgressInfo], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> Path:
    """Download, verify (SHA-256), and extract the llama.cpp runtime for `backend`."""
    os_key = current_os_key()
    archives = RUNTIME_ARCHIVES.get((os_key, backend))
    if not archives:
        raise DownloadError(f"Unsupported platform/backend combination: {os_key}-{backend}")

    rt_dir = get_runtime_dir(backend)
    archive_dir = rt_dir / "_archives"
    ready_marker = rt_dir / ".ready"
    ready_marker.unlink(missing_ok=True)

    # 1. Resumable download with SHA-256 verification
    download_artifacts(
        [ra.artifact for ra in archives],
        archive_dir,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        check_hash=True,
    )

    if cancel_event and cancel_event.is_set():
        raise DownloadCancelled("Runtime installation cancelled")

    # 2. Extract archives into bin/ (and cudart/ on Linux CUDA)
    for ra in archives:
        if cancel_event and cancel_event.is_set():
            raise DownloadCancelled("Runtime installation cancelled")
        archive_path = archive_dir / ra.artifact.rel_path
        dest_dir = rt_dir / ra.target_subdir
        if status_callback:
            status_callback(f"Extracting {ra.artifact.rel_path.name}…")
        if archive_path.name.endswith(".tar.gz"):
            _safe_extract_tar(archive_path, dest_dir, strip_components=ra.strip_components)
        elif archive_path.name.endswith(".zip"):
            _safe_extract_zip(archive_path, dest_dir, strip_components=ra.strip_components)
        else:
            raise DownloadError(f"Unsupported archive format: {archive_path.name}")

    # Ensure llama-server is executable on POSIX
    server_bin = rt_dir / "bin" / server_binary_name()
    if sys.platform != "win32" and server_bin.exists():
        mode = server_bin.stat().st_mode
        os.chmod(server_bin, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # Mark ready and clean up archives to save disk space
    ready_marker.write_text(f"{os_key}-{backend}-{RUNTIME_VERSION}\n", encoding="utf-8")
    shutil.rmtree(archive_dir, ignore_errors=True)
    return rt_dir

