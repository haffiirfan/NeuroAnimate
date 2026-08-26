# 🧠 NeuroAnimate — OrchestraGen Pipeline

### Memory-Orchestrated Multimodal Synthesis for 3D-Styled Imagery & Hyper-Realistic Portrait Animation

[![Paper](https://img.shields.io/badge/Paper-Springer%20Nature%20MMSJ-blue)](https://www.springer.com/journal/530)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Kaggle%20T4×2-orange)](https://kaggle.com)

> **📄 Paper Status:** Under peer review at *Multimedia Systems* (MMSJ), Springer Nature.

---

## 🎯 Overview

**NeuroAnimate** is a six-model generative AI pipeline that transforms a simple text prompt into a fully animated, hyper-realistic portrait video. The core contribution is **OrchestraGen** — a **Dynamic Memory Orchestration (DMO)** strategy that enables models totalling **~60.1 GB** in weights to run sequentially within **32 GB** of combined VRAM on consumer-grade hardware (2× NVIDIA Tesla T4), without quantisation or quality loss.

### The Pipeline

```
Text Prompt
    │
    ▼
┌──────────────────────┐
│  1. Mistral 7B       │  GPU 1  │  Prompt Enhancement
│     (14.5 GB FP16)   │         │  → SDXL-optimised description
└──────────┬───────────┘
           │ ── VRAM CLEAR ──
           ▼
┌──────────────────────┐
│  2. SDXL 1.0 Base    │  GPU 0  │  Text-to-Image (768×768)
│     (6.9 GB FP16)    │         │
└──────────┬───────────┘
           │ ── VRAM CLEAR ──
           ▼
┌──────────────────────┐
│  3. SDXL 1.0 Refiner │  GPU 1  │  Detail Enhancement
│     (6.2 GB FP16)    │         │
└──────────┬───────────┘
           │ ── FULL VRAM CLEAR ──
           ▼
┌──────────────────────┬──────────────────────┐
│  4a. LivePortrait    │  4b. 2D Body Anim.   │  PARALLEL
│      (GPU 0, 3 GB)   │      (CPU only)       │
│      Face animation  │      Breathing/sway   │
└──────────┬───────────┴──────────┬───────────┘
           │ ── VRAM CLEAR ──     │
           ▼                      ▼
┌─────────────────────────────────────────────┐
│  5. Shoulder-Split Compositor (CPU)          │
│     Feathered blend + colour correction     │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  6. Real-ESRGAN 1.5× Upscale (GPU 0 + GPU 1)│
│     Dual-GPU parallel frame processing       │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
            Animated Video
```

---

## 🔬 Key Contribution: Dynamic Memory Orchestration (DMO)

| Metric | Without DMO | With DMO |
|:--|:--:|:--:|
| **Peak VRAM required** | 60.1 GB | **14.5 GB** |
| **Hardware needed** | 4× A100 (80 GB each) | **2× T4 (16 GB each)** |
| **Quantisation** | Not needed | **Not needed** |
| **Quality loss** | None | **None** |

DMO enforces a strict **load → infer → teardown → clear** lifecycle for each model. Between stages, `torch.cuda.empty_cache()` and `gc.collect()` are called on every GPU to reclaim all VRAM before the next model loads. See [`src/memory_orchestrator.py`](src/memory_orchestrator.py) for the full implementation.

---

## 📁 Project Structure

```
NeuroAnimate/
├── app.py                         # Gradio web interface
├── CITATION.md                    # BibTeX citation for the paper
├── LICENSE                        # MIT License
├── README.md                      # This file
├── requirements.txt               # Pinned dependencies
│
├── configs/
│   └── pipeline_config.yaml       # All tunable pipeline parameters
│
├── src/
│   ├── __init__.py                # Package exports
│   ├── memory_orchestrator.py     # 🔑 DMO — the novel contribution
│   ├── prompt_enhancer.py         # Stage 1: Mistral 7B prompt enhancement
│   ├── image_generator.py         # Stage 2-3: SDXL Base + Refiner
│   ├── liveportrait_runner.py     # Stage 4a: Face animation (GPU 0)
│   ├── body_animator.py           # Stage 4b: 2D body animation (CPU)
│   ├── compositor.py              # Stage 5: Face + body blending
│   ├── upscaler.py                # Stage 6: Real-ESRGAN dual-GPU upscale
│   └── pipeline.py                # Master pipeline orchestrator
│
├── notebooks/
│   └── neuroanimate.ipynb         # Original Kaggle notebook
│
├── outputs/                       # Sample output videos
│   └── .gitkeep
│
└── assets/                        # Architecture diagrams
    └── .gitkeep
```

---

## 🚀 Quick Start (Kaggle)

This pipeline is designed for **Kaggle Notebooks** with **GPU T4 ×2** accelerator.

### 1. Create a new Kaggle Notebook
- Go to [kaggle.com/notebooks](https://www.kaggle.com/code) → **New Notebook**
- Under **Settings → Accelerator**, select **GPU T4 ×2**

### 2. Clone this repository
```python
!git clone https://github.com/haffiirfan/NeuroAnimate-Multimodal-Synthesis-for-3D-styled-imagery-hyper-realistic-Shorts.git
%cd NeuroAnimate-Multimodal-Synthesis-for-3D-styled-imagery-hyper-realistic-Shorts
```

### 3. Install dependencies
```python
!pip install -r requirements.txt
```

### 4. Launch the Gradio UI
```python
from app import demo
demo.launch(share=True)
```

### 5. Or use the pipeline programmatically
```python
from src.pipeline import NeuroAnimatePipeline

pipeline = NeuroAnimatePipeline()

# Stage 1: Enhance prompt
enhanced = pipeline.enhance("A warrior in golden armor", mode="Gaming")

# Stage 2-3: Generate portrait
images = pipeline.generate_images(enhanced, num_images=1, seed=42)

# Stage 4-6: Animate into video
video_path, status = pipeline.generate_video("driving_video.mp4")
print(status)
```

---

## 🛠️ Models Used

| # | Model | Parameters | VRAM (FP16) | Purpose |
|:--:|:--|:--:|:--:|:--|
| 1 | [Nous-Hermes-2-Mistral-7B-DPO](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO) | 7.2B | 14.5 GB | Prompt enhancement |
| 2 | [SDXL 1.0 Base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | 3.5B | 6.9 GB | Text-to-image generation |
| 3 | [SDXL 1.0 Refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) | 3.1B | 6.2 GB | Detail enhancement |
| 4 | [LivePortrait](https://github.com/KwaiVGI/LivePortrait) | — | 3.0 GB | Facial motion transfer |
| 5 | InsightFace | — | 0.5 GB | Face detection & alignment |
| 6 | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | 16.7M | 0.1 GB | 1.5× super-resolution |
| | **Total** | | **~60.1 GB** | **(but only 14.5 GB peak with DMO)** |

---

## 📊 Performance (Kaggle T4 ×2)

| Stage | Time | Hardware |
|:--|:--:|:--|
| Prompt Enhancement (Mistral 7B) | ~15s | GPU 1 |
| Image Generation (SDXL Base) | ~25s | GPU 0 |
| Image Refinement (SDXL Refiner) | ~30s | GPU 1 |
| Face Animation (LivePortrait) | ~45s | GPU 0 |
| Body Animation (2D Procedural) | ~30s | CPU (parallel) |
| Compositing | ~10s | CPU |
| Super-Resolution (ESRGAN) | ~60s | GPU 0 + GPU 1 |
| **Total Pipeline** | **~3.5 min** | |

---

## 📄 Paper

**OrchestraGen: Memory-Orchestrated Multimodal Synthesis for 3D-Styled Imagery and Hyper-Realistic Portrait Animation**

- **Author:** Haffi Irfan
- **Journal:** Multimedia Systems (MMSJ), Springer Nature
- **Status:** Under Peer Review
- **Year:** 2026

See [CITATION.md](CITATION.md) for the BibTeX entry.

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait) — Face animation
- [Stability AI](https://stability.ai/) — Stable Diffusion XL
- [NousResearch](https://nousresearch.com/) — Nous-Hermes-2-Mistral-7B
- [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — Super-resolution
- [InsightFace](https://insightface.ai/) — Face detection and recognition
