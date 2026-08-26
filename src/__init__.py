"""
NeuroAnimate — OrchestraGen Pipeline
=====================================
Memory-Orchestrated Multimodal Synthesis for 3D-Styled Imagery
& Hyper-Realistic Portrait Animation.

A six-model generative AI pipeline that chains Mistral-7B, SDXL 1.0
(Base + Refiner), InsightFace, LivePortrait, and Real-ESRGAN within
32 GB of combined VRAM using Dynamic Memory Orchestration (DMO).

Paper: Under peer review at Springer Nature — Multimedia Systems (MMSJ)
Author: Haffi Irfan (haffiirfan@gmail.com)
"""

__version__ = "1.0.0"
__author__ = "Haffi Irfan"

from .pipeline import NeuroAnimatePipeline
from .memory_orchestrator import clear_gpu_memory, clear_gpu
from .prompt_enhancer import PromptEnhancer
from .image_generator import ImageGenerator
from .liveportrait_runner import normalize_video, run_liveportrait
from .body_animator import run_body_animation
from .compositor import composite_videos
from .upscaler import run_esrgan
