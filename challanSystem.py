"""
Traffic Challan System
======================
Models:
  - anpr_detector.pt     → detects license plate regions
  - helmet_detector.pt   → detects helmet / no-helmet
  - seatbelt_detector.pt → detects seatbelt / no-seatbelt
  - yolov8n.pt           → vehicle detection (auto-downloads)
  - Plate Recognizer API → reads plate text (high accuracy)

Violations that trigger a challan:
  - Motorcycle rider with no helmet
  - Car driver with no seatbelt

Supports: single image / video file / webcam

Install:
    pip install ultralytics opencv-python requests

Run:
    py challan_system.py --source image        (uses Test_images/)
    py challan_system.py --source video        (uses test_video.mp4)
    py challan_system.py --source webcam
    py challan_system.py --source image --input Test_images/myfile.png
"""

import os
import cv2
import json
import time
import requests
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — Edit these
# ══════════════════════════════════════════════════════════════════════════════
PLATE_RECOGNIZER_TOKEN = "7cc9a346f4357355ee1b7d959182aa129bccb54b"          # ← paste your token
ANPR_MODEL_PATH        = r"models/anpr_detector.pt"
HELMET_MODEL_PATH      = r"models/helmet.pt"
SEATBELT_MODEL_PATH    = r"models/seatbelt.pt"
OUTPUT_DIR             = r"challans"
CONF_THRESH            = 0.3

# Default input paths
DEFAULT_IMAGE          = r"Test_images/Screenshot 2026-03-22 212858.png"
DEFAULT_VIDEO          = r"test_video.mp4"

# PlateRecognizer region — "in" for India
PR_REGION              = "in"

# COCO vehicle class IDs (yolov8n)
VEHICLE_CLASSES        = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# ── Drawing colors (BGR) ──────────────────────────────────────────────────────
CLR_VEHICLE    = (219, 152, 52)    # blue
CLR_PLATE      = (0, 215, 255)     # yellow
CLR_VIOLATION  = (0, 0, 220)       # red
CLR_OK         = (50, 200, 50)     # green
CLR_LABEL_BG   = (0, 140, 255)     # orange
CLR_WHITE      = (255, 255, 255)
CLR_BLACK      = (0, 0, 0)
# ══════════════════════════════════════════════════════════════════════════════


# ── Plate Recognizer API call ─────────────────────────────────────────────────
def read_plate_platerecognizer(crop_bgr):
    """Send crop to Plate Recognizer API, return plate text string."""
    if PLATE_RECOGNIZER_TOKEN == "YOUR_API_TOKEN_HERE":
        return "NO_API_KEY"
    _, img_encoded = cv2.imencode(".jpg", crop_bgr)
    try:
        response = requests.post(
            "https://api.platerecognizer.com/v1/plate-reader/",
            files={"upload": ("plate.jpg", img_encoded.tobytes(), "image/jpeg")},
            data={"regions": PR_REGION},
            headers={"Authorization": f"Token {PLATE_RECOGNIZER_TOKEN}"},
            timeout=10,
        )
        data = response.json()
        if data.get("results"):
            return data["results"][0]["plate"].upper()
        return "UNREAD"
    except Exception as e:
        print(f"  [PlateRec ERROR] {e}")
        return "API_ERROR"


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_box(img, x1, y1, x2, y2, color, thickness=2):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def draw_label(img, text, x, y, bg=CLR_LABEL_BG, fg=CLR_WHITE, scale=0.55, thick=1):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    pad = 4
    cv2.rectangle(img, (x, y - th - pad * 2), (x + tw + pad * 2, y + bl), bg, -1)
    cv2.putText(img, text, (x + pad, y - pad), font, scale, fg, thick, cv2.LINE_AA)


def draw_violation_banner(img, x1, y1, violations):
    """Draw stacked red violation tags above a vehicle box."""
    for i, v in enumerate(violations):
        yy = y1 - 24 * (len(violations) - i)
        draw_label(img, f"⚠ {v}", x1, yy, bg=CLR_VIOLATION)


# ── IoU / spatial helpers ─────────────────────────────────────────────────────
def box_center_in(inner, outer):
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def overlap_ratio(a, b):
    ax1,ay1,ax2,ay2 = a;  bx1,by1,bx2,by2 = b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1)
    inter = iw*ih
    if inter == 0: return 0.0
    areaA = (ax2-ax1)*(ay2-ay1)
    return inter/areaA   # ratio of A that overlaps B


# ── Challan logger ────────────────────────────────────────────────────────────
challan_log = []

def log_challan(plate, violations, frame_id=None):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "plate":      plate,
        "violations": violations,
        "frame":      frame_id,
    }
    challan_log.append(entry)
    tag = f"[CHALLAN #{len(challan_log)}]"
    print(f"{tag} Plate={plate}  Violations={violations}")
    return entry


# ══════════════════════════════════════════════════════════════════════════════
#  CORE DETECTION FUNCTION  (runs on one frame)
# ══════════════════════════════════════════════════════════════════════════════
def process_frame(frame, models, frame_id=None, save_crops=True):
    """
    Run all models on one frame. Returns annotated frame + list of challans.
    """
    vehicle_det, anpr_det, helmet_det, seatbelt_det = models
    canvas = frame.copy()
    H, W   = frame.shape[:2]
    challans_this_frame = []

    def clamp(x1,y1,x2,y2):
        return max(0,int(x1)), max(0,int(y1)), min(W,int(x2)), min(H,int(y2))

    # ── 1. Detect vehicles ────────────────────────────────────────────────────
    veh_res = vehicle_det.predict(source=frame, conf=0.3, verbose=False)[0]
    vehicles = []
    for box in (veh_res.boxes or []):
        cid = int(box.cls[0])
        if cid not in VEHICLE_CLASSES: continue
        x1,y1,x2,y2 = clamp(*box.xyxy[0].tolist())
        vehicles.append({
            "cls":  VEHICLE_CLASSES[cid],
            "conf": float(box.conf[0]),
            "box":  [x1,y1,x2,y2],
            "plates": [],
            "violations": [],
        })

    # ── 2. Detect plates (ANPR) ───────────────────────────────────────────────
    plate_res = anpr_det.predict(source=frame, conf=CONF_THRESH, verbose=False)[0]
    plates = []
    for box in (plate_res.boxes or []):
        x1,y1,x2,y2 = clamp(*box.xyxy[0].tolist())
        if x2<=x1 or y2<=y1: continue
        crop = frame[y1:y2, x1:x2]
        text = read_plate_platerecognizer(crop)
        plates.append({"box":[x1,y1,x2,y2], "conf":float(box.conf[0]), "text":text, "crop":crop})

    # ── 3. Detect helmets ─────────────────────────────────────────────────────
    helm_res  = helmet_det.predict(source=frame, conf=CONF_THRESH, verbose=False)[0]
    no_helmets = []
    for box in (helm_res.boxes or []):
        label = helm_res.names[int(box.cls[0])].lower()
        if "no" in label or "without" in label or label == "no_helmet":
            x1,y1,x2,y2 = clamp(*box.xyxy[0].tolist())
            no_helmets.append([x1,y1,x2,y2])

    # ── 4. Detect seatbelts ───────────────────────────────────────────────────
    seat_res   = seatbelt_det.predict(source=frame, conf=CONF_THRESH, verbose=False)[0]
    no_seatbelts = []
    for box in (seat_res.boxes or []):
        label = seat_res.names[int(box.cls[0])].lower()
        if "no" in label or "without" in label or label == "no_seatbelt":
            x1,y1,x2,y2 = clamp(*box.xyxy[0].tolist())
            no_seatbelts.append([x1,y1,x2,y2])

    # ── 5. Associate plates → vehicles ────────────────────────────────────────
    for plate in plates:
        matched = None
        for veh in vehicles:
            if box_center_in(plate["box"], veh["box"]):
                matched = veh; break
        if matched:
            matched["plates"].append(plate)
        else:
            # Standalone plate (no vehicle matched) — still draw it
            vehicles.append({
                "cls": "Vehicle", "conf": plate["conf"],
                "box": plate["box"], "plates": [plate], "violations": []
            })

    # ── 6. Check violations per vehicle ──────────────────────────────────────
    for veh in vehicles:
        vx1,vy1,vx2,vy2 = veh["box"]
        # Expand vehicle box upward to catch helmet on rider's head
        exp_box = [vx1, max(0, vy1 - int((vy2-vy1)*0.4)), vx2, vy2]

        if veh["cls"] == "Motorcycle":
            for nh in no_helmets:
                if (box_center_in(nh, exp_box)
                        or box_center_in(nh, veh["box"])
                        or overlap_ratio(nh, exp_box) > 0.15):
                    if "No Helmet" not in veh["violations"]:
                        veh["violations"].append("No Helmet")

        if veh["cls"] in ("Car", "Bus", "Truck"):
            for ns in no_seatbelts:
                if (box_center_in(ns, exp_box)
                        or overlap_ratio(ns, exp_box) > 0.15):
                    if "No Seatbelt" not in veh["violations"]:
                        veh["violations"].append("No Seatbelt")

        # Log challan if violations found
        # Issue challan even without plate (plate = UNKNOWN) so no violation is missed
        if veh["violations"]:
            plate_text = veh["plates"][0]["text"] if veh["plates"] else "UNKNOWN"
            entry = log_challan(plate_text, veh["violations"], frame_id)
            challans_this_frame.append(entry)

            # Save crop for evidence
            if save_crops and veh["plates"]:
                Path(OUTPUT_DIR, "evidence").mkdir(parents=True, exist_ok=True)
                ts  = datetime.now().strftime("%H%M%S_%f")[:10]
                ev  = os.path.join(OUTPUT_DIR, "evidence", f"{plate_text}_{ts}.jpg")
                px1,py1,px2,py2 = veh["plates"][0]["box"]
                cv2.imwrite(ev, frame[py1:py2, px1:px2])

    # ── 7. Draw everything ────────────────────────────────────────────────────
    for veh in vehicles:
        vx1,vy1,vx2,vy2 = veh["box"]
        has_violation = len(veh["violations"]) > 0

        # Vehicle box
        vbox_color = CLR_VIOLATION if has_violation else CLR_VEHICLE
        draw_box(canvas, vx1, vy1, vx2, vy2, vbox_color, thickness=2)

        # Vehicle type label
        draw_label(canvas, veh["cls"], vx1, vy1, bg=vbox_color)

        # Violation banners
        if has_violation:
            draw_violation_banner(canvas, vx1, vy1, veh["violations"])

        # Plate boxes + orange label
        for plate in veh["plates"]:
            px1,py1,px2,py2 = plate["box"]
            draw_box(canvas, px1, py1, px2, py2, CLR_PLATE, thickness=2)
            label = f"{plate['text']}  {veh['cls']}  {plate['conf']:.2f}"
            draw_label(canvas, label, px1, py1, bg=CLR_LABEL_BG)

    # Draw no-helmet / no-seatbelt boxes
    for nh in no_helmets:
        draw_box(canvas, *nh, CLR_VIOLATION, 1)
        draw_label(canvas, "No Helmet", nh[0], nh[1], bg=CLR_VIOLATION, scale=0.45)
    for ns in no_seatbelts:
        draw_box(canvas, *ns, CLR_VIOLATION, 1)
        draw_label(canvas, "No Seatbelt", ns[0], ns[1], bg=CLR_VIOLATION, scale=0.45)

    # Challan count overlay (top-left)
    total = len(challan_log)
    cv2.rectangle(canvas, (0, 0), (260, 36), CLR_BLACK, -1)
    cv2.putText(canvas, f"Total Challans: {total}", (8, 25),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, CLR_WHITE, 1, cv2.LINE_AA)

    return canvas, challans_this_frame


# ══════════════════════════════════════════════════════════════════════════════
#  INPUT MODES
# ══════════════════════════════════════════════════════════════════════════════
def run_image(models, input_path):
    frame = cv2.imread(input_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read: {input_path}")
    print(f"[INFO] Processing image: {input_path}")
    result, _ = process_frame(frame, models, frame_id=0)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, Path(input_path).stem + "_challan_result.jpg")
    cv2.imwrite(out, result)
    print(f"[DONE] Saved → {out}")
    cv2.imshow("Challan System", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(models, input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, Path(input_path).stem + "_challan_result.mp4")
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    frame_id = 0
    print(f"[INFO] Processing video: {input_path}")
    while True:
        ret, frame = cap.read()
        if not ret: break
        result, _ = process_frame(frame, models, frame_id=frame_id, save_crops=True)
        writer.write(result)
        cv2.imshow("Challan System — press Q to quit", result)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
        frame_id += 1
        if frame_id % 30 == 0:
            print(f"  Frame {frame_id}  |  Challans so far: {len(challan_log)}")
    cap.release(); writer.release(); cv2.destroyAllWindows()
    print(f"[DONE] Video saved → {out_path}")


def run_webcam(models):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")
    print("[INFO] Webcam started — press Q to quit")
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        result, _ = process_frame(frame, models, frame_id=frame_id, save_crops=False)
        cv2.imshow("Challan System — press Q to quit", result)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
        frame_id += 1
    cap.release(); cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Traffic Challan System")
    parser.add_argument("--source", choices=["image","video","webcam"], default="image")
    parser.add_argument("--input",  default=None, help="Path to image or video file")
    args = parser.parse_args()

    print("[INFO] Loading models...")
    vehicle_det  = YOLO("yolov8n.pt")
    anpr_det     = YOLO(ANPR_MODEL_PATH)
    helmet_det   = YOLO(HELMET_MODEL_PATH)
    seatbelt_det = YOLO(SEATBELT_MODEL_PATH)
    models = (vehicle_det, anpr_det, helmet_det, seatbelt_det)
    print("[INFO] All models loaded.\n")

    if args.source == "image":
        run_image(models, args.input or DEFAULT_IMAGE)
    elif args.source == "video":
        run_video(models, args.input or DEFAULT_VIDEO)
    elif args.source == "webcam":
        run_webcam(models)

    # Save challan log to JSON
    if challan_log:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(OUTPUT_DIR, "challan_log.json")
        with open(log_path, "w") as f:
            json.dump(challan_log, f, indent=2)
        print(f"\n[LOG] {len(challan_log)} challan(s) saved → {log_path}")
    else:
        print("\n[LOG] No violations detected.")


if __name__ == "__main__":
    main()