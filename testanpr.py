"""
ANPR Pipeline — Vehicle Detection + Plate Detection + Tesseract OCR
--------------------------------------------------------------------
Produces annotated image with:
  - Blue boxes  : vehicles (YOLOv8n pretrained)
  - Orange label: plate text + detection conf + vehicle type

Install deps:
    pip install pytesseract ultralytics opencv-python
    + Tesseract-OCR engine: https://github.com/UB-Mannheim/tesseract/wiki

Run:
    py testanpr.py
"""

import os
import cv2
import numpy as np
import pytesseract
from pathlib import Path
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────
ANPR_MODEL_PATH  = r"models/anpr_detector.pt"
IMAGE_PATH       = r"Test_images/Screenshot 2026-03-22 212858.png"
CONF_THRESH      = 0.3
OUTPUT_DIR       = r"anpr_output"
TESSERACT_CMD    = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# COCO vehicle class IDs (YOLOv8n pretrained)
VEHICLE_CLASSES  = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Colors (BGR)
COLOR_VEHICLE    = (255, 160, 50)    # blue-ish
COLOR_PLATE      = (0, 200, 255)     # yellow
COLOR_LABEL_BG   = (0, 140, 255)     # orange
COLOR_TEXT       = (255, 255, 255)   # white
# ─────────────────────────────────────────────────────────────────────────────

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def preprocess_plate(crop_bgr):
    """Upscale + CLAHE + sharpen for better OCR on Indian plates."""
    h, w    = crop_bgr.shape[:2]
    scale   = 80 / max(h, 1)
    new_w   = max(int(w * scale), 1)
    resized = cv2.resize(crop_bgr, (new_w, 80), interpolation=cv2.INTER_CUBIC)
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    blurred  = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1)
    sharp    = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    return sharp


def read_plate_tesseract(crop_bgr):
    """Run Tesseract OCR on a plate crop."""
    proc   = preprocess_plate(crop_bgr)
    config = r"--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text   = pytesseract.image_to_string(proc, config=config)
    text   = text.strip().replace(" ", "").replace("\n", "")
    return text if text else "UNKNOWN"


def draw_label(img, text, pos, bg_color=COLOR_LABEL_BG, text_color=COLOR_TEXT, font_scale=0.55, thickness=1):
    """Draw a filled rectangle label with text."""
    font    = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y    = pos
    pad     = 4
    # Background rect
    cv2.rectangle(img, (x, y - th - pad * 2), (x + tw + pad * 2, y + baseline), bg_color, -1)
    # Text
    cv2.putText(img, text, (x + pad, y - pad), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_box(img, x1, y1, x2, y2, color, thickness=2):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def iou(boxA, boxB):
    """Intersection over Union between two boxes [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(areaA + areaB - interArea)


def plate_center_in_box(plate_box, veh_box):
    """Check if plate center falls inside vehicle box."""
    cx = (plate_box[0] + plate_box[2]) / 2
    cy = (plate_box[1] + plate_box[3]) / 2
    return veh_box[0] <= cx <= veh_box[2] and veh_box[1] <= cy <= veh_box[3]


def main():
    # ── Load models ───────────────────────────────────────────────────────────
    print("[INFO] Loading vehicle detector (YOLOv8n)...")
    vehicle_detector = YOLO("yolov8n.pt")   # auto-downloads if not present

    print("[INFO] Loading ANPR plate detector...")
    plate_detector = YOLO(ANPR_MODEL_PATH)

    # ── Load image ────────────────────────────────────────────────────────────
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")
    print(f"[INFO] Image: {IMAGE_PATH}  shape={image.shape}\n")

    canvas = image.copy()
    H, W   = image.shape[:2]

    # ── Vehicle detection ─────────────────────────────────────────────────────
    veh_results = vehicle_detector.predict(source=image, conf=0.3, verbose=False)[0]
    vehicles    = []
    for box in (veh_results.boxes or []):
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:
            continue
        conf  = float(box.conf[0])
        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
        x1=max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
        vehicles.append({"box": [x1,y1,x2,y2], "cls": VEHICLE_CLASSES[cls_id], "conf": conf, "plates": []})

    print(f"[INFO] Vehicles detected : {len(vehicles)}")

    # ── Plate detection ───────────────────────────────────────────────────────
    plate_results = plate_detector.predict(source=image, conf=CONF_THRESH, verbose=False)[0]
    plates        = []
    for box in (plate_results.boxes or []):
        conf  = float(box.conf[0])
        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
        x1=max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop      = image[y1:y2, x1:x2]
        plate_txt = read_plate_tesseract(crop)
        plates.append({"box": [x1,y1,x2,y2], "conf": conf, "text": plate_txt, "crop": crop})

    print(f"[INFO] Plates detected   : {len(plates)}")

    # ── Associate plates → vehicles ───────────────────────────────────────────
    for plate in plates:
        best_veh = None
        best_iou = 0.0
        for veh in vehicles:
            if plate_center_in_box(plate["box"], veh["box"]):
                score = iou(plate["box"], veh["box"])
                if score > best_iou or best_veh is None:
                    best_iou = score
                    best_veh = veh
        if best_veh:
            best_veh["plates"].append(plate)
        else:
            # No vehicle matched — draw plate standalone
            vehicles.append({"box": plate["box"], "cls": "Vehicle", "conf": plate["conf"], "plates": [plate]})

    # ── Draw everything ───────────────────────────────────────────────────────
    for veh in vehicles:
        vx1,vy1,vx2,vy2 = veh["box"]

        # Draw vehicle box (blue)
        draw_box(canvas, vx1, vy1, vx2, vy2, COLOR_VEHICLE, thickness=2)

        # Vehicle type label (top-left of box)
        draw_label(canvas, veh["cls"], (vx1, vy1), bg_color=COLOR_VEHICLE)

        for plate in veh["plates"]:
            px1,py1,px2,py2 = plate["box"]

            # Draw plate box (yellow)
            draw_box(canvas, px1, py1, px2, py2, COLOR_PLATE, thickness=2)

            # Orange label: PlateText  | VehicleType | Conf
            label = f"{plate['text']}  {veh['cls']}  {plate['conf']:.2f}"
            draw_label(canvas, label, (px1, py1), bg_color=COLOR_LABEL_BG)

            # Console output
            print(f"  [{veh['cls']}] Plate: {plate['text']:15s}  conf={plate['conf']:.2f}")

    # ── Save output ───────────────────────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    base_name  = Path(IMAGE_PATH).stem
    out_path   = os.path.join(OUTPUT_DIR, f"{base_name}_result.jpg")
    cv2.imwrite(out_path, canvas)
    print(f"\n[DONE] Saved → {out_path}")


if __name__ == "__main__":
    main()