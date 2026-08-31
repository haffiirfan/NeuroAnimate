# Citation

If you use this code or the OrchestraGen pipeline in your research, please cite:

```bibtex
@article{irfan2026orchestragen,
  title     = {OrchestraGen: Memory-Orchestrated Multimodal Synthesis for
               3D-Styled Imagery and Hyper-Realistic Portrait Animation},
  author    = {Haffi Irfan, Dr. Muhammad Saleem},
  journal   = {Multimedia Systems},
  publisher = {Springer Nature},
  year      = {2026},
  note      = {Under Review}
}
```

## Paper Abstract

We present **OrchestraGen**, a six-model generative AI pipeline that synthesises
3D-styled portrait imagery from text prompts and animates the result into
hyper-realistic video — all within 32 GB of combined GPU VRAM on consumer-grade
hardware (2× NVIDIA Tesla T4). The key contribution is **Dynamic Memory
Orchestration (DMO)**, a sequential load–infer–unload strategy that enables
models totalling ~60.1 GB in weights to share limited GPU memory without
quantisation or quality loss. The pipeline chains Mistral-7B for prompt
enhancement, Stable Diffusion XL (Base + Refiner) for image generation,
LivePortrait for facial motion transfer, a procedural 2D body animator, and
Real-ESRGAN for super-resolution upscaling. Experiments on Kaggle T4×2 hardware
demonstrate that DMO reduces peak VRAM from 60.1 GB (simultaneous) to 14.5 GB
(sequential peak) while maintaining generation quality.
