"""
Prompt Enhancement Module — Mistral 7B (GPU 1)

Enhances raw user prompts into SDXL-optimised descriptions using the
Nous-Hermes-2-Mistral-7B-DPO language model. Supports three creative
modes: Photorealism, Graphic Design, and Gaming.

The enhanced prompt is capped at 77 tokens to respect the CLIP text
encoder limit in Stable Diffusion XL.

OrchestraGen Note:
    Mistral 7B is loaded on GPU 1 in FP16 (~14.5 GB VRAM). After prompt
    generation, the model is explicitly deleted and VRAM is cleared before
    the next stage (SDXL Base on GPU 0).
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .memory_orchestrator import clear_gpu


# ── System Prompts for Each Creative Mode ───

SYSTEM_PROMPT_PHOTOREALISM = (
    "You are an SDXL 1.0 prompt engineer for photorealistic DSLR/Canon images. "
    "Enhance the user prompt into ~52–68 tokens. "
    "Keep the original subject Exactly as given, just enhance it properly. "
    "Must Add: ultra-detailed, camera type (DSLR/mirrorless), lens (mm & f-stop), angle, "
    "time of night or day, lighting, realistic textures, shadows and atmosphere. "
    "Write in 2–3 fluent sentences. "
    "Never use cartoon, anime, painting, illustration styles and storytelling."
)

SYSTEM_PROMPT_GRAPHIC = (
    "You are an elite SDXL prompt enhancer and professional graphic/logo design director. "
    "Rewrite and expand the user's idea into ~40–60 tokens (2–3 short sentences) while "
    "keeping the exact subject. Include: usage contexts, logo variants, vector-friendly "
    "scalable geometry & safe spacing, color palette with HEX, 3D material + lighting cues, "
    "photorealistic mockups, typography pairing, high-contrast modern timeless aesthetic, "
    "SVG-ready composition. Always use the brand product view."
)

SYSTEM_PROMPT_GAMING = (
    "You are an SDXL prompt engineer and AAA game character designer. "
    "Rewrite the user's idea into ~50–60 tokens while keeping the same subject. "
    "Always include: 8k ultra-sharp details, half upper body, Unreal Engine 5 render, "
    "RTX lighting, PBR materials, polygonal hard edges, dark 3D colors, and next-gen shaders, "
    "with tactical armor, sharp metallic textures, detailed weapon. "
    "Focus on sharp edges, high-resolution textures, and the gritty realism of games. "
    "Write in 2–3 fluent sentences emphasising AAA in-game cutscene style, not movie realism."
)


class PromptEnhancer:
    """
    Wraps Mistral 7B for single-shot prompt enhancement.

    Attributes:
        model_name (str): HuggingFace model identifier.
        device_id (int): Target GPU index (default: 1).
    """

    def __init__(self, model_name="NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
                 device_id=1):
        self.model_name = model_name
        self.device_id = device_id

    def _get_system_prompt(self, mode, user_prompt):
        """Selects the appropriate system prompt based on creative mode."""
        mode_lower = str(mode).strip().lower()

        if mode_lower in ("graphic designer", "graphic", "designer"):
            template = SYSTEM_PROMPT_GRAPHIC
        elif mode_lower in ("gaming", "game", "concept artist"):
            template = SYSTEM_PROMPT_GAMING
        else:
            template = SYSTEM_PROMPT_PHOTOREALISM

        return (
            f"{template}\n\n"
            f"User Prompt: {user_prompt}\n"
            f"Enhanced Prompt:"
        )

    def enhance(self, user_prompt, mode="Photorealism", bypass=False):
        """
        Enhances a user prompt using Mistral 7B.

        Args:
            user_prompt (str): Raw user description.
            mode (str): Creative mode — 'Photorealism', 'Graphic Designer', or 'Gaming'.
            bypass (bool): If True, returns the original prompt unchanged.

        Returns:
            tuple: (enhanced_prompt: str, elapsed_seconds: float)
        """
        if bypass:
            return user_prompt, 0.0

        try:
            clear_gpu()
            t0 = time.time()

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map={"": self.device_id},
                trust_remote_code=True,
            ).eval()

            system_prompt = self._get_system_prompt(mode, user_prompt)

            inputs = tokenizer(
                system_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=180,
            )
            inputs = {
                k: v.to(torch.device(f"cuda:{self.device_id}"))
                for k, v in inputs.items()
            }

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=90,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[-1]
            gen_tokens = outputs[0][input_len:]
            enhanced = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            # Enforce CLIP 77-token limit
            words = enhanced.split()
            if len(words) > 77:
                enhanced = " ".join(words[:77])

            elapsed = time.time() - t0

            # ── DMO: Explicit model teardown ──
            del model, tokenizer, inputs, outputs
            clear_gpu()

            return enhanced, elapsed

        except Exception as e:
            clear_gpu()
            raise RuntimeError(f"Prompt enhancement failed: {e}") from e
