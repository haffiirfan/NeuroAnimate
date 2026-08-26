"""
Image Generation Module — SDXL 1.0 Base (GPU 0) + Refiner (GPU 1)
==================================================================

Generates high-fidelity portrait images using the two-stage Stable Diffusion
XL pipeline: Base model produces the initial 768×768 image, then the Refiner
applies detail enhancement at low denoising strength.

OrchestraGen Note:
    SDXL Base (~6.9 GB) loads on GPU 0 after Mistral 7B has been unloaded
    from GPU 1. After base generation, GPU 0 is cleared and the Refiner
    (~6.2 GB) is loaded on GPU 1. After refinement, BOTH GPUs are fully
    cleared before the video animation stages begin.
"""

import os
import math
import time
import torch
from datetime import datetime
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from .memory_orchestrator import clear_gpu


class ImageGenerator:
    """
    Two-stage SDXL image generator with sequential GPU orchestration.

    Attributes:
        base_model (str): HuggingFace ID for SDXL Base.
        refiner_model (str): HuggingFace ID for SDXL Refiner.
        base_device (int): GPU index for base generation (default: 0).
        refiner_device (int): GPU index for refinement (default: 1).
    """

    def __init__(
        self,
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        refiner_model="stabilityai/stable-diffusion-xl-refiner-1.0",
        base_device=0,
        refiner_device=1,
    ):
        self.base_model = base_model
        self.refiner_model = refiner_model
        self.base_device = base_device
        self.refiner_device = refiner_device

    def generate(self, prompt, num_images=2, seed=42, output_dir="refined_images"):
        """
        Generates refined portrait images.

        Args:
            prompt (str): The (enhanced) text prompt.
            num_images (int): Number of images to generate (1–4).
            seed (int): Random seed for reproducibility.
            output_dir (str): Directory to save refined PNGs.

        Returns:
            tuple: (list[PIL.Image], dict[str, float]) — images and timing info.
        """
        timings = {}

        try:
            clear_gpu()

            # ── Stage 1: SDXL Base on GPU 0 ──────────────────────────────
            torch.cuda.set_device(self.base_device)
            pipe_base = StableDiffusionXLPipeline.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            ).to(f"cuda:{self.base_device}")

            base_images = []
            t0 = time.time()
            for i in range(num_images):
                generator = torch.Generator(f"cuda:{self.base_device}").manual_seed(
                    seed + i
                )
                image = pipe_base(
                    prompt,
                    num_inference_steps=35,
                    guidance_scale=8,
                    height=768,
                    width=768,
                    generator=generator,
                ).images[0]
                base_images.append(image)

            timings["sdxl_base"] = time.time() - t0

            # ── DMO: Teardown Base, clear GPU 0 ─────────────────────────
            del pipe_base
            clear_gpu()

            # ── Stage 2: SDXL Refiner on GPU 1 ──────────────────────────
            torch.cuda.set_device(self.refiner_device)
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.refiner_model,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            ).to(f"cuda:{self.refiner_device}")

            refined_images = []
            os.makedirs(output_dir, exist_ok=True)
            t0 = time.time()

            for idx, img in enumerate(base_images):
                refined = refiner(
                    prompt=prompt,
                    image=img,
                    num_inference_steps=105,
                    guidance_scale=8,
                    strength=0.25,
                ).images[0]
                refined_images.append(refined)

                filename = os.path.join(
                    output_dir,
                    f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx + 1}.png",
                )
                refined.convert("RGB").save(filename)

            timings["sdxl_refiner"] = time.time() - t0

            # ── DMO: Full teardown — both GPUs cleared before video ──────
            del refiner
            clear_gpu()

            return refined_images, timings

        except Exception as e:
            clear_gpu()
            raise RuntimeError(f"Image generation failed: {e}") from e

    @staticmethod
    def create_grid(images):
        """
        Arranges multiple images into a square grid for display.

        Args:
            images (list[PIL.Image]): List of generated images.

        Returns:
            PIL.Image: Grid image.
        """
        if len(images) <= 1:
            return images[0] if images else None

        cols = math.ceil(math.sqrt(len(images)))
        rows = math.ceil(len(images) / cols)
        width, height = images[0].size
        grid = Image.new("RGB", (cols * width, rows * height))
        for i, img in enumerate(images):
            grid.paste(img, (i % cols * width, i // cols * height))
        return grid
