# Citation

If you use this code or the OrchestraGen pipeline in your research, please cite:

```bibtex
@article{irfan2026orchestragen,
  title     = {OrchestraGen: Memory-Orchestrated Multimodal Synthesis for
               3D-Styled Imagery and Hyper-Realistic Portrait Animation},
  author    = {Irfan, Haffi},
  journal   = {Multimedia Systems},
  publisher = {Springer Nature},
  year      = {2026},
  note      = {Under Review}
}
```

## Paper Abstract

Recent advances in multimedia generative artificial intelligence have
been driven by large-scale vision–language diffusion models and increasing
specialization in computer vision. Although the performance of individual
models continues to improve, deploying complex multi-model AI pipelines in
real-world applications remains challenging due to resource conflicts, memory
constraints, version incompatibilities, and limited robustness. Moreover, existing
text-to-avatar generation systems typically rely on high-end GPUs, which limits
their scalability on consumer-grade hardware. We propose OrchestraGen, a
multi-model generative pipeline that transforms textual prompts into fully
animated portrait videos. OrchestraGen uses LLM-based prompt enhancement,
dual-stage diffusion synthesis, 3D-aware portrait animation through custom
body motion via facial retargeting, and super-resolution enhancement on
constrained dual-GPU hardware. Dynamic Memory Orchestration enables
reliable execution of 60.1 GB of heterogeneous model weights on a 32 GB
dual-T4 configuration, achieving a 1.88× system memory scaling factor while
maintaining peak single-GPU occupancy at 13.8 GB. Heterogeneous GPU
CPU parallelism and dual-GPU parallel frame enhancement collectively reduce
animation latency by 33.1%, while three-mode LLM prompt enhancement
achieves a CLIP alignment score of 0.35.
