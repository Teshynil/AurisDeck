"""Isolated hardware device discovery and device-selection resolution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Literal

_CUDA_PROBE_SCRIPT = """
import json
import subprocess

devices = []
try:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        def _to_int(val):
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return None

        for line in proc.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            name = parts[1] or f"CUDA:{idx}"
            total_mb = _to_int(parts[2]) if len(parts) > 2 else None
            free_mb = _to_int(parts[3]) if len(parts) > 3 else None
            devices.append(
                {
                    "id": f"cuda:{idx}",
                    "kind": "cuda",
                    "index": idx,
                    "name": name,
                    "vram_total_mb": total_mb,
                    "vram_free_mb": free_mb,
                }
            )
except Exception:
    devices = []

if not devices:
    try:
        import torch

        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                free_mb = total_mb = None
                try:
                    free_b, total_b = torch.cuda.mem_get_info(idx)
                    free_mb = int(free_b // (1024 * 1024))
                    total_mb = int(total_b // (1024 * 1024))
                except Exception:
                    pass
                devices.append(
                    {
                        "id": f"cuda:{idx}",
                        "kind": "cuda",
                        "index": idx,
                        "name": torch.cuda.get_device_name(idx),
                        "vram_total_mb": total_mb,
                        "vram_free_mb": free_mb,
                    }
                )
    except Exception:
        devices = []

if not devices:
    try:
        import ctranslate2

        count = int(ctranslate2.get_cuda_device_count())
        for idx in range(count):
            devices.append(
                {
                    "id": f"cuda:{idx}",
                    "kind": "cuda",
                    "index": idx,
                    "name": f"CUDA Device {idx}",
                    "vram_total_mb": None,
                    "vram_free_mb": None,
                }
            )
    except Exception:
        devices = []

print(json.dumps(devices))
""".strip()


@dataclass(slots=True)
class DeviceInfo:
    """Descriptor for an available compute device (CUDA GPU or CPU)."""

    id: str
    kind: Literal["cuda", "cpu"]
    index: int | None
    name: str
    vram_total_mb: int | None = None
    vram_free_mb: int | None = None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_probe_output(raw_stdout: str) -> list[DeviceInfo]:
    text = raw_stdout.strip()
    if not text:
        return []

    records: list[dict[str, Any]] = []
    try:
        parsed = json.loads(text)
    except ValueError:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except ValueError:
                return []
            if isinstance(item, dict):
                records.append(item)
    else:
        if isinstance(parsed, list):
            records = [item for item in parsed if isinstance(item, dict)]
        elif isinstance(parsed, dict):
            records = [parsed]

    devices: list[DeviceInfo] = []
    for pos, record in enumerate(records):
        index = _optional_int(record.get("index"))
        if index is None:
            index = pos
        device_id = str(record.get("id") or f"cuda:{index}")
        name = str(record.get("name") or f"CUDA:{index}")
        devices.append(
            DeviceInfo(
                id=device_id,
                kind="cuda",
                index=index,
                name=name,
                vram_total_mb=_optional_int(record.get("vram_total_mb")),
                vram_free_mb=_optional_int(record.get("vram_free_mb")),
            )
        )
    return devices


def discover_devices() -> list[DeviceInfo]:
    """Discover CUDA GPUs in an isolated subprocess and always append CPU."""
    cuda_devices: list[DeviceInfo] = []
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CUDA_PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode == 0:
            cuda_devices = _parse_probe_output(proc.stdout)
    except (subprocess.SubprocessError, OSError, ValueError):
        cuda_devices = []

    cpu_threads = os.cpu_count() or 1
    cpu_device = DeviceInfo(
        id="cpu",
        kind="cpu",
        index=None,
        name=f"CPU ({cpu_threads} threads)",
        vram_total_mb=None,
        vram_free_mb=None,
    )
    return [*cuda_devices, cpu_device]


def resolve_allowed_devices(
    gpus_spec: str | list[int],
    discovered: list[DeviceInfo],
    cpu_fallback: bool = True,
) -> list[str]:
    """Resolve a GPU specification ('auto', 'cpu', or index list) to device IDs."""
    if gpus_spec == "cpu":
        return ["cpu"]

    if gpus_spec == "auto":
        cuda_ids = [device.id for device in discovered if device.kind == "cuda"]
        if cuda_ids:
            return cuda_ids
        return ["cpu"] if cpu_fallback else []

    if isinstance(gpus_spec, list):
        available = {
            device.index: device.id
            for device in discovered
            if device.kind == "cuda" and device.index is not None
        }
        matched = [available[idx] for idx in dict.fromkeys(gpus_spec) if idx in available]
        if matched:
            return matched
        return ["cpu"] if cpu_fallback else []

    raise ValueError(f"Unsupported gpus_spec: {gpus_spec!r}")
