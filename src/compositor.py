"""
Face + Body Compositor — Shoulder-Split Blending

Merges the LivePortrait face animation (upper region) with the 2D body
animation (lower region) using a feathered shoulder-split mask with
per-channel colour correction at the seam boundary.

The split point is automatically estimated from the source portrait's
face detection: chin position + 1.6× face height. An 80-pixel Gaussian
fade prevents visible seams between the two animation sources.
"""

import os
import json
import subprocess
import numpy as np
import cv2


def ffprobe_info(path):
    """Extracts video stream metadata via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", path,
        ],
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    return next((s for s in streams if s["codec_type"] == "video"), {})


def composite_videos(lp_video, body_video, source_image, trimmed_video,
                     work_dir="/kaggle/working", log=print):
    """
    Composites face (LivePortrait) and body (2D animation) videos using
    a shoulder-split mask with colour correction.

    Args:
        lp_video (str): Path to the LivePortrait face animation.
        body_video (str): Path to the 2D body animation.
        source_image (str): Path to the original source portrait.
        trimmed_video (str): Path to the driving video (for audio track).
        work_dir (str): Working directory for output files.
        log: Callable for progress output.

    Returns:
        str: Path to the composited output video.
    """
    log("🎬 Compositing face + body (shoulder-split) ...")

    FINAL = os.path.join(work_dir, "final_output.mp4")

    # ── Determine output dimensions from body video ───
    bi = ffprobe_info(body_video)
    BW = int(bi["width"])
    BH = int(bi["height"])
    num, den = bi.get("r_frame_rate", "25/1").split("/")
    FPS_C = float(num) / float(den)

    # ── Detect face in source for split-point estimation ───
    src_r = cv2.resize(cv2.imread(source_image), (BW, BH))
    gray_s = cv2.cvtColor(src_r, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    fx = fy = fw2 = fh = None
    for sf, mn, msz in [(1.05, 6, (60, 60)), (1.05, 4, (50, 50)), (1.1, 3, (40, 40))]:
        detections = face_cascade.detectMultiScale(
            gray_s, scaleFactor=sf, minNeighbors=mn, minSize=msz
        )
        if len(detections) > 0:
            fx, fy, fw2, fh = max(detections, key=lambda r: r[2] * r[3])
            break

    if fx is not None:
        chin_y = fy + fh
        SPLIT_Y = min(int(chin_y + fh * 1.6), int(BH * 0.80))
        log(f"   Face: y={fy} h={fh} chin={chin_y} → split y={SPLIT_Y}")
    else:
        SPLIT_Y = int(BH * 0.62)
        log(f"   ⚠️ No face — fallback split y={SPLIT_Y}")

    # ── Build feathered blend masks ──
    FADE = 80
    lp_mask = np.zeros((BH, BW), dtype=np.float32)
    for y in range(BH):
        if y < SPLIT_Y - FADE:
            lp_mask[y, :] = 1.0
        elif y < SPLIT_Y + FADE:
            lp_mask[y, :] = (SPLIT_Y + FADE - y) / (2.0 * FADE)
    lp_mask = cv2.GaussianBlur(lp_mask, (1, 101), 30)
    lp_m3 = np.stack([lp_mask] * 3, axis=-1)
    body_m3 = 1.0 - lp_m3

    # ── Per-channel colour correction at the seam ──
    def peek_frame(path):
        """Extracts the first frame from a video for colour analysis."""
        proc = subprocess.Popen(
            [
                "ffmpeg", "-i", path, "-frames:v", "1", "-f", "rawvideo",
                "-pix_fmt", "bgr24", "-vf", f"scale={BW}:{BH}", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        raw = proc.stdout.read(BW * BH * 3)
        proc.wait()
        if len(raw) == BW * BH * 3:
            return np.frombuffer(raw, dtype=np.uint8).reshape((BH, BW, 3)).astype("float32")
        return None

    lp_f0 = peek_frame(lp_video)
    body_f0 = peek_frame(body_video)
    cc_gain = np.ones(3, dtype="float32")
    if lp_f0 is not None and body_f0 is not None:
        st_t = max(0, SPLIT_Y - 30)
        st_b = min(BH, SPLIT_Y + 30)
        for c in range(3):
            lm = float(lp_f0[st_t:st_b, :, c].mean()) + 1e-3
            bm = float(body_f0[st_t:st_b, :, c].mean()) + 1e-3
            cc_gain[c] = lm / bm
        cc_gain = np.clip(cc_gain, 0.80, 1.25)
        log(f"   Colour-correction gain (BGR): {cc_gain.round(3).tolist()}")

    # ── Frame-by-frame compositing via ffmpeg pipes ───
    def open_pipe(path):
        return subprocess.Popen(
            [
                "ffmpeg", "-i", path, "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-vf", f"scale={BW}:{BH}", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def read_frame(pipe):
        raw = pipe.stdout.read(BW * BH * 3)
        if len(raw) < BW * BH * 3:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape((BH, BW, 3)).copy()

    fp = open_pipe(lp_video)
    bp = open_pipe(body_video)
    fout = subprocess.Popen(
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{BW}x{BH}", "-r", str(FPS_C), "-i", "pipe:0",
            "-i", trimmed_video, "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p", "-shortest", FINAL,
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    idx = 0
    while True:
        fb = read_frame(fp)
        bb = read_frame(bp)
        if fb is None or bb is None:
            break
        fb_f = fb.astype("float32")
        bb_f = np.clip(
            bb.astype("float32") * cc_gain[np.newaxis, np.newaxis, :], 0, 255
        )
        comp = fb_f * lp_m3 + bb_f * body_m3
        fout.stdin.write(np.clip(comp, 0, 255).astype("uint8").tobytes())
        idx += 1

    fp.stdout.close(); fp.wait()
    bp.stdout.close(); bp.wait()
    fout.stdin.close(); fout.wait()

    log(f"✅ Composited {idx} frames → {os.path.getsize(FINAL) / 1e6:.1f} MB")
    return FINAL
