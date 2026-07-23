#!/usr/bin/env python3
"""Small helpers for piping MuJoCo RGB frames directly to FFmpeg."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np


def resolve_ffmpeg_exe() -> Path:
    env_path = os.environ.get("MYOEXO_FFMPEG_EXE") or os.environ.get("IMAGEIO_FFMPEG_EXE")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    local = shutil.which("ffmpeg")
    if local:
        return Path(local)
    bundled = Path(
        "/home/lzn/miniconda3/envs/myoassist-mjwarp/lib/python3.11/site-packages/"
        "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
    )
    if bundled.exists():
        return bundled
    raise RuntimeError("ffmpeg not found. Set MYOEXO_FFMPEG_EXE to an ffmpeg binary.")


def ffmpeg_encoders(ffmpeg: Path | None = None) -> set[str]:
    exe = resolve_ffmpeg_exe() if ffmpeg is None else Path(ffmpeg)
    result = subprocess.run(
        [str(exe), "-hide_banner", "-encoders"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def choose_h264_encoder(ffmpeg: Path, requested: str = "auto") -> str:
    requested = str(requested or "auto").strip()
    encoders = ffmpeg_encoders(ffmpeg)
    if requested == "auto":
        if "h264_nvenc" in encoders:
            return "h264_nvenc"
        if "libx264" in encoders:
            return "libx264"
        raise RuntimeError(f"{ffmpeg} has neither h264_nvenc nor libx264")
    if requested not in encoders:
        raise RuntimeError(f"{ffmpeg} does not provide requested encoder {requested!r}")
    return requested


def open_rgb_h264_writer(
    out_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    encoder: str | None = None,
    cq: int = 23,
) -> tuple[subprocess.Popen[bytes], dict[str, Any]]:
    ffmpeg = resolve_ffmpeg_exe()
    selected = choose_h264_encoder(ffmpeg, encoder or os.environ.get("MYOEXO_H264_ENCODER", "auto"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg),
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{int(width)}x{int(height)}",
        "-r",
        str(int(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        selected,
    ]
    if selected == "h264_nvenc":
        cmd += ["-preset", "p1", "-cq", str(int(cq))]
    elif selected == "libx264":
        cmd += ["-preset", "veryfast", "-crf", str(int(cq))]
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc, {"ffmpeg": str(ffmpeg), "encoder": selected, "cmd": cmd}


def write_rgb_frame(proc: subprocess.Popen[bytes], image: np.ndarray) -> None:
    if proc.stdin is None:
        raise RuntimeError("ffmpeg stdin is closed")
    proc.stdin.write(np.ascontiguousarray(image, dtype=np.uint8).tobytes())


def close_rgb_h264_writer(proc: subprocess.Popen[bytes]) -> None:
    if proc.stdin is not None:
        proc.stdin.close()
    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
    return_code = proc.wait()
    if return_code != 0:
        tail = stderr.strip()[-2000:]
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {tail}")
