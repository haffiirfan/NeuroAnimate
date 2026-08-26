"""
LivePortrait Runner — Face Animation (GPU 0)
=============================================

Handles video preprocessing (normalisation to 512×768 @ 25fps) and
face-driven animation using the KwaiVGI LivePortrait model.

LivePortrait transfers facial motion from a driving video onto a source
portrait image, producing a video where the generated face moves naturally
according to the driver's expressions and head pose.

OrchestraGen Note:
    LivePortrait runs on GPU 0 in parallel with 2D body animation on CPU.
    After completion, GPU 0 is cleared before the Real-ESRGAN upscaling stage.
"""

import os
import glob
import shutil
import subprocess
from huggingface_hub import snapshot_download


# ── Default Paths (Kaggle Environment) ───────────────────────────────────────
WORK = "/kaggle/working"
LP_DIR = "/kaggle/working/LivePortrait"


def setup_liveportrait(work_dir=WORK, lp_dir=LP_DIR, log=print):
    """
    Clones the LivePortrait repository and downloads pretrained weights
    if not already present.

    Args:
        work_dir (str): Working directory.
        lp_dir (str): LivePortrait installation directory.
        log: Callable for progress output.
    """
    if not os.path.exists(lp_dir):
        subprocess.run(
            ["git", "clone", "https://github.com/KwaiVGI/LivePortrait", lp_dir],
            check=True,
        )
        log("✅ LivePortrait cloned")
    else:
        log("✅ LivePortrait present")

    weights_dir = os.path.join(lp_dir, "pretrained_weights", "liveportrait")
    if not os.path.exists(weights_dir):
        os.chdir(lp_dir)
        snapshot_download(
            repo_id="KwaiVGI/LivePortrait",
            local_dir="pretrained_weights",
            ignore_patterns=["*.git*", "README.md", "docs/*"],
        )
        log("✅ LivePortrait weights downloaded")
    else:
        log("✅ LivePortrait weights present")

    os.chdir(work_dir)


def normalize_video(source_image_path, driving_video_path,
                    work_dir=WORK, log=print):
    """
    Prepares inputs for the animation pipeline:
      - Copies the source portrait to a standard location
      - Rescales and pads the driving video to 512×768 @ 25fps

    Args:
        source_image_path (str): Path to the source portrait image.
        driving_video_path (str): Path to the raw driving video.
        work_dir (str): Working directory for output files.
        log: Callable for progress output.

    Returns:
        tuple: (normalised_image_path, trimmed_video_path)
    """
    log("📥 Normalising inputs ...")
    src = os.path.join(work_dir, "my_image.png")
    trm = os.path.join(work_dir, "driving_trimmed.mp4")

    shutil.copy(source_image_path, src)

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", driving_video_path,
            "-vf", (
                "scale=512:768:force_original_aspect_ratio=decrease,"
                "pad=512:768:(ow-iw)/2:(oh-ih)/2:color=black"
            ),
            "-r", "25", "-c:v", "libx264", "-crf", "18", "-preset", "fast", trm,
        ],
        check=True,
        stderr=subprocess.DEVNULL,
    )

    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0", trm,
        ],
        capture_output=True,
        text=True,
    )
    log(f"✅ Driving video ready — {result.stdout.strip()} frames @ 512×768 25fps")
    return src, trm


def run_liveportrait(source_image, trimmed_video, motion_multiplier=0.65,
                     lp_dir=LP_DIR, work_dir=WORK, log=print):
    """
    Runs LivePortrait face animation on GPU 0.

    Patches the flag_lip_zero configuration to False (prevents lip freezing
    artefact) and runs inference with relative motion transfer and pasteback.

    Args:
        source_image (str): Path to the normalised source portrait.
        trimmed_video (str): Path to the preprocessed driving video.
        motion_multiplier (float): Scales the intensity of facial motion (default: 0.65).
        lp_dir (str): LivePortrait installation directory.
        work_dir (str): Working directory.
        log: Callable for progress output.

    Returns:
        str: Path to the generated face animation video.
    """
    log(f"🎭 [GPU 0] Running LivePortrait (motion×{motion_multiplier}) ...")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    # ── Patch lip_zero flag (one-time) ───────────────────────────────────
    patch_marker = os.path.join(lp_dir, ".lip_zero_patched")
    if not os.path.exists(patch_marker):
        for cfg in [
            os.path.join(lp_dir, "src", "config", "argument_config.py"),
            os.path.join(lp_dir, "src", "config", "inference_config.py"),
        ]:
            if os.path.exists(cfg):
                with open(cfg, "r") as fh:
                    content = fh.read()
                patched = content.replace(
                    "flag_lip_zero: bool = True", "flag_lip_zero: bool = False"
                ).replace("flag_lip_zero=True", "flag_lip_zero=False")
                if patched != content:
                    with open(cfg, "w") as fh:
                        fh.write(patched)
                    log(f"   ✅ Patched flag_lip_zero→False in {os.path.basename(cfg)}")
        open(patch_marker, "w").close()

    # ── Run inference ────────────────────────────────────────────────────
    os.chdir(lp_dir)
    subprocess.run(
        [
            "python", "inference.py",
            "-s", source_image,
            "-d", trimmed_video,
            "--flag-relative-motion", "--flag-pasteback", "--flag-do-crop",
            "--driving-multiplier", str(motion_multiplier),
            "--scale", "3.0",
        ],
        check=True,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    os.chdir(work_dir)

    # ── Locate output video ──────────────────────────────────────────────
    candidates = [
        v
        for v in glob.glob(os.path.join(lp_dir, "animations", "**", "*.mp4"), recursive=True)
        if "concat" not in os.path.basename(v)
    ]
    if not candidates:
        candidates = glob.glob(
            os.path.join(lp_dir, "animations", "**", "*.mp4"), recursive=True
        )
    lp_output = max(candidates, key=os.path.getmtime)
    log(f"✅ Face animation → {os.path.basename(lp_output)}")
    return lp_output
