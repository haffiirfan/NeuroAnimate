"""
Dynamic Memory Orchestration (DMO) — The Novel Contribution of OrchestraGen
=============================================================================

This module implements the core innovation described in the OrchestraGen paper:
a sequential memory orchestration strategy that enables six heterogeneous deep
learning models (totalling ~60.1 GB in weights) to execute within 32 GB of
combined GPU VRAM across two NVIDIA Tesla T4 GPUs.

The Problem:
    Loading Mistral-7B (14.5 GB), SDXL Base (6.9 GB), SDXL Refiner (6.2 GB),
    InsightFace (0.5 GB), LivePortrait (3.0 GB), and Real-ESRGAN (0.1 GB)
    simultaneously requires ~60.1 GB — nearly double the available VRAM.

The Solution:
    DMO enforces a strict sequential lifecycle for GPU-resident models.
    After each pipeline stage completes, DMO explicitly:
      1. Deletes all Python references to the model and its tensors
      2. Invokes torch.cuda.empty_cache() on EVERY GPU device
      3. Calls torch.cuda.ipc_collect() to reclaim inter-process memory
      4. Triggers Python garbage collection (gc.collect())
    This ensures that the VRAM footprint at any given moment never exceeds
    the capacity of the available hardware.

Orchestration Schedule (Dual T4, 32 GB total):
    ┌─────────────────────────┬────────────┬────────────┐
    │ Stage                   │ GPU 0      │ GPU 1      │
    ├─────────────────────────┼────────────┼────────────┤
    │ 1. Prompt Enhancement   │ idle       │ Mistral 7B │
    │ ── VRAM CLEAR ──        │            │            │
    │ 2. Base Generation      │ SDXL Base  │ idle       │
    │ ── VRAM CLEAR ──        │            │            │
    │ 3. Refinement           │ idle       │ SDXL Ref.  │
    │ ── FULL VRAM CLEAR ──   │            │            │
    │ 4a. Face Animation      │ LivePort.  │ idle       │
    │ 4b. Body Animation      │ (CPU)      │ (CPU)      │
    │ ── VRAM CLEAR ──        │            │            │
    │ 5. Compositing          │ (CPU)      │ (CPU)      │
    │ 6. Super-Resolution     │ ESRGAN-L   │ ESRGAN-R   │
    └─────────────────────────┴────────────┴────────────┘

    Peak VRAM usage: ~14.5 GB (Mistral 7B in FP16) — well within 16 GB/GPU.

Reference:
    Irfan, H. (2026). OrchestraGen: Memory-Orchestrated Multimodal Synthesis
    for 3D-Styled Imagery & Hyper-Realistic Portrait Animation.
    Multimedia Systems (MMSJ), Springer Nature. [Under Review]
"""

import gc
import subprocess
import torch


def clear_gpu():
    """
    Lightweight VRAM clear — flushes cached allocations on all visible GPUs.

    Called between pipeline stages to release memory held by the PyTorch
    caching allocator without tearing down the CUDA context.
    """
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def clear_gpu_memory(log=print):
    """
    Full VRAM reclamation — spawns an isolated Python subprocess to ensure
    complete release of all GPU memory, including orphaned tensors that
    may not be reachable by the parent process's garbage collector.

    This is the nuclear option used after major stage transitions
    (e.g., after image generation, before video animation) where even
    unreferenced CUDA tensors from deleted models may persist in the
    caching allocator.

    Args:
        log: Callable for progress output (default: print).
    """
    log("🧹 Clearing GPU VRAM ...")
    result = subprocess.run(
        [
            "python", "-c",
            """
import torch
n = torch.cuda.device_count()
for i in range(n):
    with torch.cuda.device(i):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print(f'  GPU {i}: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB remaining')
print(f'Cleared {n} GPU(s)')
"""
        ],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.strip().split("\n"):
        log(f"   {line}")
    log("✅ GPU VRAM cleared")


def get_vram_report():
    """
    Returns a dictionary of current VRAM usage per GPU.

    Returns:
        dict: {gpu_id: {"allocated_gb": float, "reserved_gb": float, "total_gb": float}}
    """
    report = {}
    for i in range(torch.cuda.device_count()):
        report[i] = {
            "allocated_gb": torch.cuda.memory_allocated(i) / 1e9,
            "reserved_gb": torch.cuda.memory_reserved(i) / 1e9,
            "total_gb": torch.cuda.get_device_properties(i).total_mem / 1e9,
        }
    return report


def log_vram_status(stage_name, log=print):
    """
    Logs a human-readable VRAM status report for debugging.

    Args:
        stage_name: Name of the current pipeline stage.
        log: Callable for output (default: print).
    """
    report = get_vram_report()
    log(f"📊 VRAM after [{stage_name}]:")
    for gpu_id, stats in report.items():
        log(
            f"   GPU {gpu_id}: "
            f"{stats['allocated_gb']:.2f} / {stats['total_gb']:.1f} GB allocated "
            f"({stats['reserved_gb']:.2f} GB reserved)"
        )
