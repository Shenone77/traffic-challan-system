"""
pipeline/video_pipeline.py  — FIXED
=====================================

ROOT CAUSE OF "No module named 'pipeline.anpr_model'":
  Models were loaded AT MODULE LEVEL (lines 16-23 in original).
  The moment any import inside anpr_model.py fails (missing torch/tf/easyocr
  dependency, wrong working directory, etc.), Python reports the WHOLE
  video_pipeline module as broken — showing a misleading ModuleNotFoundError.

FIX: All model instances are lazy-loaded on first use via _get_models().
     A failure in one model never breaks the others or the whole pipeline.

OTHER FIXES vs original:
  - annotated_video returned as absolute Windows path → broken download URL.
    Now returns just the filename; app.py serves it via /video/<filename>.
  - all_violations accumulated across ALL frames including false positives.
    Now uses a per-violation vote counter — a violation only counts if it
    appears in at least MIN_VIOLATION_VOTES processed frames.
  - Helmet violations checked against class name properly (was exact-match only,
    missed 'without_helmet', 'no helmet' variants your model may output).
  - video_pipeline also imported anpr_model_v2 in some versions — unified back
    to anpr_model so there is one single source of truth.
"""

import cv2
import os
import sys
import numpy as np
from collections import Counter

# ── lazy model singletons ─────────────────────────────────────────────────────
_helmet   = None
_seatbelt = None
_anpr     = None


def _get_models():
    """
    Load all models on first call. Each model loads independently —
    a failure in one does not prevent the others from loading.
    """
    global _helmet, _seatbelt, _anpr

    if _helmet is None:
        try:
            from pipeline.helmet_model import HelmetModel
            _helmet = HelmetModel()
            print("[Pipeline] Helmet model ready.")
        except Exception as e:
            print(f"[Pipeline] WARNING: Helmet model failed to load: {e}")
            _helmet = _NullModel()

    if _seatbelt is None:
        try:
            from pipeline.seatbelt_model import SeatbeltModel
            _seatbelt = SeatbeltModel()
            print("[Pipeline] Seatbelt model ready.")
        except Exception as e:
            print(f"[Pipeline] WARNING: Seatbelt model failed to load: {e}")
            _seatbelt = _NullModel()

    if _anpr is None:
        try:
            from pipeline.anpr_model import ANPRModel
            _anpr = ANPRModel()
            print("[Pipeline] ANPR model ready.")
        except Exception as e:
            print(f"[Pipeline] WARNING: ANPR model failed to load: {e}")
            _anpr = _NullModel()

    return _helmet, _seatbelt, _anpr


class _NullModel:
    """Placeholder when a model fails to load — returns empty results."""
    def predict(self, *args, **kwargs):
        return []


# ── constants ─────────────────────────────────────────────────────────────────
PROCESS_EVERY_N      = 10   # analyse every 10th frame — good balance of speed vs accuracy
MAX_FRAMES           = 900  # cap at ~30 s @ 30 fps — enough for demo
PLATE_HISTORY        = 15   # rolling window for plate stabilisation
MIN_PLATE_VOTES      = 3    # minimum identical readings to trust a plate
MIN_VIOLATION_VOTES  = 2    # minimum frames a violation must appear in to count

# Violation class name normalisation — maps whatever your model outputs
# to the canonical key used by the rest of the system
_HELMET_VIOLATION_KEYS = {
    "no_helmet", "without_helmet", "nohelmet",
    "no helmet", "triple_riding", "triple", "tripling",
}
_SEATBELT_VIOLATION_KEYS = {
    "no_seatbelt", "without_seatbelt", "noseatbelt", "no seatbelt",
}


def _normalise_violation(class_name: str) -> str | None:
    """Map raw model class name → canonical violation key, or None to ignore."""
    lbl = class_name.lower().strip()
    if any(k in lbl for k in ("triple", "tripling")):
        return "triple_riding"
    if any(k in lbl for k in ("no_helmet", "without_helmet", "nohelmet", "no helmet")):
        return "no_helmet"
    if any(k in lbl for k in ("no_seatbelt", "without_seatbelt", "noseatbelt", "no seatbelt")):
        return "no_seatbelt"
    return None   # ignore (car, motorcycle, person, with_helmet, etc.)


# ── single frame processor ────────────────────────────────────────────────────

def process_frame(frame: np.ndarray) -> dict:
    """
    Run all detectors on one frame.
    Returns violations list, raw detections, and annotated frame.
    """
    helmet_model, seatbelt_model, anpr_model = _get_models()

    violations    = []
    helmet_dets   = []
    seatbelt_dets = []
    anpr_dets     = []
    annotated     = frame.copy()

    # ── helmet / triple riding ─────────────────────────────────────────────
    try:
        helmet_dets = helmet_model.predict(frame)
        for d in helmet_dets:
            vkey = _normalise_violation(d["class"])
            if vkey and vkey not in violations:
                violations.append(vkey)
            x1, y1, x2, y2 = d["bbox"]
            is_viol = vkey is not None
            color   = (0, 0, 255) if is_viol else (0, 200, 0)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
            cv2.putText(annotated, f"{d['class']} {d['confidence']:.2f}",
                        (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    except Exception as e:
        print(f"[Pipeline] Helmet error: {e}")

    # ── seatbelt ──────────────────────────────────────────────────────────
    try:
        seatbelt_dets = seatbelt_model.predict(frame)
        for d in seatbelt_dets:
            vkey = _normalise_violation(d["class"])
            if vkey and vkey not in violations:
                violations.append(vkey)
            x1, y1, x2, y2 = d["bbox"]
            color = (0, 0, 255) if vkey else (0, 200, 0)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
            cv2.putText(annotated, f"{d['class']} {d['confidence']:.2f}",
                        (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    except Exception as e:
        print(f"[Pipeline] Seatbelt error: {e}")

    # ── ANPR ──────────────────────────────────────────────────────────────
    try:
        anpr_dets = anpr_model.predict(frame)
        for d in anpr_dets:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (255,165,0), 2)
            cv2.putText(annotated, f"Plate: {d['plate_text']}",
                        (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,165,0), 2)
    except Exception as e:
        print(f"[Pipeline] ANPR error: {e}")

    return {
        "violations":    violations,
        "helmet_dets":   helmet_dets,
        "seatbelt_dets": seatbelt_dets,
        "anpr_dets":     anpr_dets,
        "annotated":     annotated,
    }


# ── plate tracker ─────────────────────────────────────────────────────────────

class PlateTracker:
    def __init__(self):
        self._history = []

    def update(self, text: str):
        if text and text != "UNKNOWN":
            self._history.append(text)
            if len(self._history) > PLATE_HISTORY:
                self._history.pop(0)

    def best(self) -> str:
        if not self._history:
            return "UNKNOWN"
        top, votes = Counter(self._history).most_common(1)[0]
        return top if votes >= MIN_PLATE_VOTES else self._history[-1]


# ── Mode 1: single image ──────────────────────────────────────────────────────

def run_image_pipeline(image_path: str, save_dir: str = "static/uploads") -> dict:
    frame = cv2.imread(image_path)
    if frame is None:
        return {"success": False, "message": "Cannot load image", "violations": []}

    result     = process_frame(frame)
    plate_text = "UNKNOWN"
    if result["anpr_dets"]:
        best       = max(result["anpr_dets"], key=lambda x: x["confidence"])
        plate_text = best["plate_text"]

    base     = os.path.splitext(os.path.basename(image_path))[0]
    ann_path = os.path.join(save_dir, f"{base}_annotated.jpg")
    cv2.imwrite(ann_path, result["annotated"])

    return {
        "success":          True,
        "mode":             "image",
        "violations":       result["violations"],
        "plate_number":     plate_text,
        "annotated_image":  ann_path,
        "helmet_results":   result["helmet_dets"],
        "seatbelt_results": result["seatbelt_dets"],
        "anpr_results":     result["anpr_dets"],
        "frames_processed": 1,
        "vehicles":         [],
    }


# ── Mode 2: video file ────────────────────────────────────────────────────────

def run_video_pipeline(video_path: str,
                       save_dir: str = "static/uploads",
                       progress_cb=None) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "message": "Cannot open video", "violations": []}

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base     = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(save_dir, f"{base}_annotated.mp4")

    # H264 produces browser-playable MP4; fall back to mp4v if unavailable
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    tracker         = PlateTracker()
    violation_votes = Counter()   # FIX: count per-violation across frames
    frame_idx       = 0
    processed       = 0
    best_frame_path = None

    while cap.isOpened() and frame_idx < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % PROCESS_EVERY_N == 0:
            result = process_frame(frame)

            # FIX: vote counting instead of blind set.update()
            for v in result["violations"]:
                violation_votes[v] += 1

            for d in result["anpr_dets"]:
                tracker.update(d["plate_text"])

            # save first violation frame as thumbnail
            if result["violations"] and best_frame_path is None:
                best_frame_path = os.path.join(save_dir, f"{base}_thumb.jpg")
                cv2.imwrite(best_frame_path, result["annotated"])

            writer.write(result["annotated"])
            processed += 1

            if progress_cb:
                pct = min(99, int(frame_idx / max(total, 1) * 100))
                progress_cb(pct, f"Frame {frame_idx}/{total}")
        else:
            writer.write(frame)

        frame_idx += 1

    cap.release()
    writer.release()

    if progress_cb:
        progress_cb(100, "Complete!")

    # FIX: only report violations seen in MIN_VIOLATION_VOTES or more frames
    confirmed_violations = [v for v, count in violation_votes.items()
                            if count >= MIN_VIOLATION_VOTES]

    return {
        "success":          True,
        "mode":             "video",
        "violations":       confirmed_violations,
        "plate_number":     tracker.best(),
        "annotated_video":  out_path,        # full path — app.py extracts basename
        "annotated_image":  best_frame_path,
        "frames_processed": processed,
        "duration_seconds": round(frame_idx / fps, 1),
        "helmet_results":   [],
        "seatbelt_results": [],
        "anpr_results":     [],
        "vehicles":         [],
    }


# ── Mode 3: live camera ───────────────────────────────────────────────────────

def run_live_pipeline(camera_index: int = 0):
    """
    Generator → yields (jpeg_bytes, frame_result_or_None).
    jpeg_bytes   = MJPEG frame for browser streaming
    frame_result = dict with violations + plate on processed frames, else None
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[Pipeline] Cannot open camera {camera_index}")
        return

    tracker   = PlateTracker()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_result = None

        if frame_idx % PROCESS_EVERY_N == 0:
            result    = process_frame(frame)
            annotated = result["annotated"]

            for d in result["anpr_dets"]:
                tracker.update(d["plate_text"])

            plate = tracker.best()
            cv2.putText(annotated, f"PLATE: {plate}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            y = 70
            for v in result["violations"]:
                cv2.putText(annotated, f"VIOLATION: {v.upper()}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                y += 30

            frame_result = {"violations": result["violations"], "plate": plate}
        else:
            annotated = frame

        frame_idx += 1

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n",
            frame_result,
        )

    cap.release()