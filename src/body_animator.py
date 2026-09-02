"""
2D Body Animation Module — CPU-Based Procedural Motion

Generates realistic upper-body motion (breathing, lateral sway, shoulder
lift) by applying sinusoidal displacement fields to the source portrait.

The animation runs entirely on CPU, allowing it to execute in parallel
with LivePortrait's face animation on GPU 0. The two outputs are later
merged by the compositor module.

Motion Components:
    1. Breathing — radial expansion/contraction from the estimated chest centre
    2. Lateral Sway — subtle horizontal oscillation weighted by the body mask
    3. Shoulder Lift — vertical shoulder movement synchronised with breathing
    4. Micro-jitter — autoregressive random noise for organic feel

Landmark Estimation:
    Uses OpenCV Haar cascade face detection to estimate neck, shoulder, and
    hip positions from the source image. Falls back to frame-proportion
    heuristics if no face is detected.
"""

import os
import math
import random
import subprocess
import ctypes
import cv2
import numpy as np


# ── Pixel Buffer Utilities ───

_DTYPE_MAP = {1: "uint8", 2: "uint16", 4: "float32", 8: "float64"}


def cv2np(array, target_dtype="float32"):
    """Zero-copy cast from OpenCV array to NumPy with explicit dtype."""
    src_dtype = _DTYPE_MAP[int(array.itemsize)]
    shape = tuple(int(x) for x in array.shape)
    nbytes = int(array.nbytes)
    result = np.empty(shape, dtype=src_dtype)
    ctypes.memmove(
        result.ctypes.data,
        ctypes.c_void_p(array.__array_interface__["data"][0]),
        nbytes,
    )
    return result.astype(target_dtype) if target_dtype != src_dtype else result


def run_body_animation(source_image, trimmed_video, work_dir="/kaggle/working",
                       log=print):
    """
    Generates 2D sinusoidal body animation from a static source portrait.

    Args:
        source_image (str): Path to the source portrait image.
        trimmed_video (str): Path to the driving video (used for frame count/FPS).
        work_dir (str): Working directory for output files.
        log: Callable for progress output.

    Returns:
        str: Path to the body animation video.
    """
    log("🕺 [CPU] 2D body animation (breathing + sway + shoulder lift) ...")

    BODY_VIDEO = os.path.join(work_dir, "body_animation.mp4")
    RAW_FRAMES = os.path.join(work_dir, "body_frames.raw")

    # ── Load and normalise source image ──
    src = cv2.imread(source_image)
    H, W = src.shape[:2]
    H = H if H % 2 == 0 else H - 1
    W = W if W % 2 == 0 else W - 1
    src = src[:H, :W]
    src_f32 = cv2np(src, "float32")

    # ── Extract driving video metadata ──
    cap = cv2.VideoCapture(trimmed_video)
    FPS = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # ── Landmark estimation via Haar cascade ──
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = None
    for scale_factor, min_neighbors, min_size in [
        (1.05, 6, (60, 60)),
        (1.05, 4, (50, 50)),
        (1.1, 3, (40, 40)),
    ]:
        detections = face_cascade.detectMultiScale(
            gray, scale_factor, min_neighbors, minSize=min_size
        )
        if len(detections) > 0:
            faces = detections
            break

    if faces is not None and len(faces) > 0:
        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        neck_y = int(fy + fh * 1.05)
        shoulder_y = neck_y + int(fh * 0.40)
        hip_y = min(H - 1, shoulder_y + int(fh * 0.90))
        spine_x = float(fx + fw / 2)
        shoulder_width = float(fw * 1.10)
        log(f"   [2D] Face ({fx},{fy},{fw},{fh}) → neck={neck_y} "
            f"shoulder={shoulder_y} hip={hip_y}")
    else:
        log("   [2D] No face detected — using frame-proportion fallbacks")
        neck_y = H // 3
        shoulder_y = neck_y + 50
        hip_y = min(H - 1, shoulder_y + 100)
        spine_x = W / 2
        shoulder_width = W * 0.35

    half_sw = max(shoulder_width / 2, 1.0)
    chest_y = int((shoulder_y + hip_y) / 2)

    # ── Weight maps ───
    ys, xs = np.mgrid[0:H, 0:W].astype("float32")
    rel_x = (xs - spine_x) / half_sw

    # Vertical body weight ramp
    body_vert = np.zeros(H, dtype="float32")
    fade_in = max(10, shoulder_y - neck_y)
    for y in range(H):
        if y <= neck_y:
            body_vert[y] = 0.0
        elif y <= shoulder_y:
            body_vert[y] = (y - neck_y) / fade_in
        elif y <= hip_y:
            body_vert[y] = 1.0
        else:
            body_vert[y] = max(0.0, 1.0 - (y - hip_y) / max(H - hip_y, 1) * 0.6)

    horiz_w = np.exp(-(rel_x ** 2) / (2 * 0.9 ** 2))
    body2d = body_vert[:, np.newaxis] * horiz_w

    # Radial displacement field from chest centre
    rad_dx = xs - spine_x
    rad_dy = ys - chest_y
    rad_r = np.sqrt(rad_dx ** 2 + rad_dy ** 2) + 1e-6
    rad_nx = rad_dx / rad_r
    rad_ny = rad_dy / rad_r

    # Shoulder-only lift mask
    shoul_vert = np.zeros(H, dtype="float32")
    top_sh = max(0, neck_y - 10)
    bot_sh = min(H - 1, shoulder_y + 60)
    for y in range(H):
        if top_sh <= y <= shoulder_y:
            shoul_vert[y] = 1.0
        elif shoulder_y < y <= bot_sh:
            shoul_vert[y] = 1.0 - (y - shoulder_y) / (bot_sh - shoulder_y)
    shoulder_mask = shoul_vert[:, np.newaxis] * horiz_w

    # Neck seam blend (LP region above, body below)
    nr = np.zeros((H, 1), dtype="float32")
    FADE_N = 30
    for y in range(H):
        if y < neck_y - FADE_N:
            nr[y] = 0.0
        elif y < neck_y + FADE_N:
            nr[y] = (y - (neck_y - FADE_N)) / (2 * FADE_N)
        else:
            nr[y] = 1.0
    m3 = np.stack([nr * np.ones((1, W), "float32")] * 3, axis=-1)
    m3i = 1.0 - m3

    # Edge guard to prevent border seams
    eh = np.clip(
        np.minimum(np.linspace(0, 1, W), np.linspace(1, 0, W)) * 6, 0, 1
    ).astype("float32")
    ev = np.clip(
        np.minimum(np.linspace(0, 1, H), np.linspace(1, 0, H)) * 6, 0, 1
    ).astype("float32")
    edge_guard = ev[:, np.newaxis] * eh[np.newaxis, :]

    # ── Animation parameters ───
    BREATH_HZ = 0.38;   BREATH_RADIAL = 3.5;  BREATH_LIFT = 2.2
    SWAY_HZ = 0.15;     SWAY_AMP = 5.0
    SHOULDER_HZ = 0.38;  SHOULDER_AMP = 2.4
    RANDOM_AMP = 0.25;  AR_ALPHA = 0.70;      RAMP_FRAMES = 20

    pr_dx = pr_dy = 0.0
    log(f"   [2D] breath={BREATH_RADIAL}px  sway={SWAY_AMP}px  "
        f"shoulder={SHOULDER_AMP}px  N={N}")

    # ── Frame-by-frame generation ──
    with open(RAW_FRAMES, "wb") as raw_f:
        for i in range(N):
            ramp = min(1.0, i / max(RAMP_FRAMES, 1))
            t = i / FPS

            breath = math.sin(2 * math.pi * BREATH_HZ * t)
            sway_sig = math.sin(2 * math.pi * SWAY_HZ * t + 0.7)
            shoul = math.sin(2 * math.pi * SHOULDER_HZ * t + 0.3)

            # Autoregressive micro-jitter
            pr_dx = AR_ALPHA * pr_dx + (1 - AR_ALPHA) * random.uniform(-1, 1) * RANDOM_AMP
            pr_dy = AR_ALPHA * pr_dy + (1 - AR_ALPHA) * random.uniform(-1, 1) * RANDOM_AMP

            # A) Breathing — radial expansion + vertical lift
            br = breath * BREATH_RADIAL * ramp
            bl = breath * BREATH_LIFT * ramp
            disp_x = rad_nx * br * body2d + pr_dx * body2d
            disp_y = rad_ny * br * body2d - bl * body2d + pr_dy * body2d

            # B) Lateral sway
            disp_x += sway_sig * SWAY_AMP * ramp * body2d

            # C) Shoulder lift
            disp_y -= shoul * SHOULDER_AMP * ramp * shoulder_mask

            # D) Apply displacement via remap
            map_x = (xs + disp_x * edge_guard).astype("float32")
            map_y = (ys + disp_y * edge_guard).astype("float32")
            warped = cv2.remap(
                src.astype("float32"), map_x, map_y,
                cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101,
            )

            out = np.clip(src_f32 * m3i + warped * m3, 0, 255).astype("uint8")
            raw_f.write(out.tobytes())

            if i % 50 == 0:
                log(f"   Body frame {i}/{N}  sway={sway_sig * SWAY_AMP * ramp:+.1f}px  "
                    f"breath={breath:+.2f}")

    # ── Encode raw frames to H.264 ──
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{W}x{H}", "-r", str(FPS), "-i", RAW_FRAMES,
            "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", BODY_VIDEO,
        ],
        check=True,
        stderr=subprocess.DEVNULL,
    )
    os.remove(RAW_FRAMES)
    log(f"✅ 2D Body animation → {os.path.getsize(BODY_VIDEO) / 1e6:.1f} MB")
    return BODY_VIDEO
