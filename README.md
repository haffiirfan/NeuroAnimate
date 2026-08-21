# Neuro Animate
### Memory-Orchestrated Multimodal Synthesis for 3D-Styled Imagery & Hyper-Realistic Portrait Animation
*Six heterogeneous AI models. 60.1 GB of weights. 32 GB of VRAM. *
---
## Overview
Neuro Animate is an end-to-end generative pipeline that transforms a single text prompt into a fully animated, super-resolved portrait video, fusing large language model prompt engineering, diffusion-based image synthesis, landmark-driven facial animation, body motion retargeting, and neural super-resolution into one unified, memory-aware execution graph.
The core engineering contribution isn't the pipeline itself, it's **how it runs.** Deploying 60.1 GB of heterogeneous model weights on hardware that only has 32 GB of VRAM should be impossible. Neuro Animate solves this through **Dynamic Memory Orchestration (DMO)**, a scheduling discipline that treats GPU memory as a managed address space,  loading, executing, and evicting models in strict sequential phases with full VRAM reclamation between stages.
Built to answer the question most multi-model demos ignore: *what happens when your pipeline exceeds your hardware?*
---
## Capability
| | What it does |
|:---|:---|
| **Prompt intelligence** | Raw user prompts are expanded and semantically enriched by Mistral-7B-Instruct (Nous-Hermes fine-tune) into detailed, diffusion-optimized scene descriptions — bridging the gap between human intent and model-ready conditioning |
| **Portrait synthesis** | SDXL Base generates a high-fidelity 1024×1024 base portrait from the enriched prompt; SDXL Refiner applies a second-pass denoising sweep for fine detail and texture coherence |
| **Facial animation** | InsightFace extracts 2D/3D facial landmarks from the generated portrait; LivePortrait re-targets expression, gaze, and head pose from a driving video onto the static face, producing temporally coherent facial motion |
| **Body motion synthesis** | Custom retargeting module uses Haar-cascade landmark estimation to synthesize upper-body motion, adding natural movement beyond facial expressions alone |
| **Super-resolution** | Real-ESRGAN upscales the final animated frames, recovering fine texture detail lost during the animation warping process |
| **Memory orchestration** | Dynamic Memory Orchestration (DMO) schedules all six models within a 32 GB dual-T4 envelope, achieving a **1.88× memory scaling factor** with peak single-GPU occupancy of just **13.8 GB** |
---
---
## Getting Started
The entire pipeline is self-contained in a single Kaggle notebook, no local installation, no dependency hell, no paid GPU instances.
```bash
# 1. Create a free Kaggle account at https://www.kaggle.com/
# 2. Upload NeuroAnimate.ipynb as a new Notebook
# 3. In the right-hand panel, set Accelerator → GPU T4 x2
# 4. Run all cells sequentially
```
The notebook will automatically download all required model weights (~60.1 GB) from HuggingFace directly into the Kaggle runtime. No manual weight management needed.
> ** Hardware Note:** This pipeline is engineered specifically for the dual-T4 Kaggle environment. Running on a single GPU or on hardware with less than 32 GB total VRAM will fail. The DMO scheduler's phase gating and memory reclamation are tuned for this exact configuration.
---
## Publication
This work is the subject of a first-author manuscript currently **under peer review** at:
> **Springer Nature—Multimedia Systems** (Impact Factor: 3.9)
>
> *Title and details available upon request.*
>
> ## Context
Neuro Animate was developed as a Final Year Project, engineered end-to-end from memory profiling and orchestration design through multi-model pipeline integration, animation synthesis, and a reproducible single-notebook deployment, to demonstrate that production-scale generative AI is not gated by hardware budget, but by how intelligently you manage the hardware you have.
---
## License
This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
---
*Built to prove that 60 GB of AI doesn't need 60 GB of GPU.*
