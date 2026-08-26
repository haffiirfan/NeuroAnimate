"""
NeuroAnimate — Gradio Web Interface
=====================================

Provides an interactive web UI for the full OrchestraGen pipeline:
  - Enter a text prompt and select a creative mode
  - Enhance the prompt using Mistral 7B
  - Generate portrait images using SDXL 1.0
  - Upload a driving video and animate the portrait
  - View the final upscaled video output

Launch:
    Run this file in a Kaggle notebook cell or locally:
        $ python app.py

The Gradio interface launches with a shareable public link.
"""

import random
import threading
import time
import traceback
import gradio as gr

from src.pipeline import NeuroAnimatePipeline
from src.memory_orchestrator import clear_gpu


# ── Initialise Pipeline ─────────────────────────────────────────────────────
pipeline = NeuroAnimatePipeline()


# ── Callback Functions ───────────────────────────────────────────────────────

def enhance_click(user_prompt, mode_choice, enhance_choice):
    """Enhances the user prompt or passes it through."""
    bypass = enhance_choice == "No"
    enhanced = pipeline.enhance(user_prompt, mode_choice, bypass)
    status_msg = (
        "✅ Using original prompt. Ready to generate images."
        if bypass
        else "✅ Prompt enhanced! Ready to generate images."
    )
    return enhanced, gr.update(visible=True), status_msg


def generate_click(enhanced_prompt, num_images, use_fixed_seed, manual_seed):
    """Generates portrait images using SDXL."""
    seed = int(manual_seed) if use_fixed_seed else random.randint(0, 2 ** 32 - 1)
    images = pipeline.generate_images(enhanced_prompt, int(num_images), seed)
    return (
        images,
        gr.update(visible=True),
        "✅ Images generated! Upload a driving video and click Animate Portrait.",
    )


def video_click(uploaded_video):
    """Runs the full video animation pipeline with live status updates."""
    if uploaded_video is None:
        yield None, "❌ Please upload a driving video first."
        return

    result_holder = [None]
    error_holder = [None]

    def _run():
        try:
            result_holder[0] = pipeline.generate_video(uploaded_video)
        except Exception:
            error_holder[0] = traceback.format_exc()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    start_time = time.time()

    while t.is_alive():
        time.sleep(1.5)
        yield None, f"⏱  Animating…  {int(time.time() - start_time)}s elapsed"

    t.join()

    if error_holder[0]:
        yield None, f"❌ Error:\n{error_holder[0][:500]}"
        return

    video_result, status = result_holder[0]
    yield video_result, status


def reset_click():
    """Resets the pipeline state and clears GPU memory."""
    pipeline.enhanced_prompt = None
    pipeline.images = None
    pipeline.timings = {}
    clear_gpu()
    return (
        None, None, None, None,
        gr.update(visible=False), gr.update(visible=False),
        "Yes", None, "🔄 Reset complete!",
    )


# ── Gradio UI Layout ────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), title="NeuroAnimate FYP Pipeline") as demo:

    gr.Markdown("# 🧠 NeuroAnimate: Text → Image → Animated Video")
    gr.Markdown(
        "**Flow:** Enhance Prompt (Mistral 7B · GPU 1) → "
        "Generate Image (SDXL · GPUs 0+1) → "
        "Animate (LivePortrait + 2D Body + ESRGAN · both GPUs)"
    )

    with gr.Row():

        # ── Left sidebar ────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            mode_choice = gr.Radio(
                choices=["Photorealism", "Graphic Designer", "Gaming"],
                value="Photorealism",
                label="Style",
            )
            enhance_choice = gr.Radio(
                choices=["Yes", "No"],
                value="Yes",
                label="Enhance Prompt?",
                info="'No' uses your original prompt directly",
            )
            num_images = gr.Slider(1, 4, value=2, step=1, label="Number of Images")
            use_fixed_seed = gr.Checkbox(value=True, label="Fixed Seed")
            manual_seed = gr.Number(value=42, label="Seed", precision=0)

            gr.Markdown("---")
            gr.Markdown("### 🎬 Driving Video")
            video_upload = gr.Video(
                label="Upload Driving Video (required for animation)",
                sources=["upload"],
                interactive=True,
            )

            gr.Markdown("---")
            with gr.Row():
                reset_btn = gr.Button("🔄 Reset")
                enhance_btn = gr.Button("✨ Enhance Prompt", variant="primary")

            generate_images_btn = gr.Button(
                "🖼️ Generate Images", variant="primary", visible=False
            )
            generate_video_btn = gr.Button(
                "🎭 Animate Portrait", variant="primary", visible=False
            )

        # ── Main panel ──────────────────────────────────────────────────
        with gr.Column(scale=2):
            user_prompt = gr.Textbox(
                lines=3,
                placeholder="Describe your portrait / subject ...",
                label="Your Prompt",
            )
            enhanced_prompt = gr.Textbox(
                lines=3, label="Enhanced Prompt", interactive=True
            )
            image_gallery = gr.Gallery(
                label="Generated Images", columns=2, height="auto"
            )
            video_output = gr.Video(label="🎥 Animated Video", height=480)

    status_text = gr.Textbox(
        label="Status",
        value="✅ Ready — choose a style, enter a prompt, and click Enhance Prompt.",
        interactive=False,
    )

    # ── Event Wiring ────────────────────────────────────────────────────
    enhance_btn.click(
        enhance_click,
        inputs=[user_prompt, mode_choice, enhance_choice],
        outputs=[enhanced_prompt, generate_images_btn, status_text],
    )
    generate_images_btn.click(
        generate_click,
        inputs=[enhanced_prompt, num_images, use_fixed_seed, manual_seed],
        outputs=[image_gallery, generate_video_btn, status_text],
    )
    generate_video_btn.click(
        video_click,
        inputs=[video_upload],
        outputs=[video_output, status_text],
    )
    reset_btn.click(
        reset_click,
        inputs=[],
        outputs=[
            user_prompt, enhanced_prompt, image_gallery, video_output,
            generate_images_btn, generate_video_btn, enhance_choice,
            video_upload, status_text,
        ],
    )


# ── Launch ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Launching NeuroAnimate Gradio UI ...")
    demo.launch(share=True, debug=True)
