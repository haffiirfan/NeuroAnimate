# NeuroAnimate
> AI-powered portrait animation pipeline — drives a static image with a reference video using LivePortrait, custom hair-motion tracking, lip-opening amplification, procedural body animation, and dual-GPU Real-ESRGAN upscaling.
---
##  Features
| Stage | What it does |
|---|---|
| **LivePortrait** | Transfers facial motion from driving video → source image (GPU 0) |
| **Hair Motion Tracker** | Optical-flow incremental hair warping keyed to facial landmarks |
| **Lip-Opening Amplifier** | Delta-based lip-region amplification for natural, expressive speech |
| **Body Animation** | Procedural breathing, shoulder sway & speech micro-motions (CPU, parallel) |
| **Face–Body Compositor** | Seamless alpha-blended merge at the neck line |
| **Real-ESRGAN Upscaler** | Dual-GPU 1.5× super-resolution for a crisp final output |

