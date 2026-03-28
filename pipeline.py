"""
Traffic Violation Detection Pipeline
Integrates: ANPR (FastALPR) + Traffic Violations (Roboflow) + Seatbelt (YOLOv5 + Keras)
"""

import os
import cv2
import numpy as np
import torch
import tensorflow as tf
from keras.models import load_model
from fast_alpr import ALPR
from roboflow import Roboflow
from decouple import config
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# FINE STRUCTURE (₹)
# ─────────────────────────────────────────────
FINE_AMOUNTS = {
    "no_helmet":     1000,
    "triple_riding": 1000,
    "wrong_lane":    500,
    "no_seatbelt":   1000,
}

# ─────────────────────────────────────────────
# SEATBELT MODEL CONFIG
# ─────────────────────────────────────────────
SEATBELT_DETECTOR_PATH  = os.getenv("SEATBELT_DETECTOR_PATH",  "models/seatbelt_best.pt")
SEATBELT_PREDICTOR_PATH = os.getenv("SEATBELT_PREDICTOR_PATH", "models/keras_model.h5")
SEATBELT_CLASS_NAMES    = {0: "No Seatbelt worn", 1: "Seatbelt Worn"}
SEATBELT_THRESHOLD      = 0.99

# ─────────────────────────────────────────────
# MODEL INITIALIZATION
# ─────────────────────────────────────────────

class ViolationPipeline:
    def __init__(self):
        logger.info("Initializing ANPR model...")
        self.alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model="cct-xs-v2-global-model",
        )

        logger.info("Initializing Traffic Violation models (Roboflow)...")
        roboflow_api_key = config("ROBOFLOW_API_KEY", default="8z59WMd6szg9jPiS3k08")
        rf = Roboflow(api_key=roboflow_api_key)
        self.helmet_model = rf.workspace().project("helmet-detection-project").version(13).model
        self.face_model   = rf.workspace().project("face-detection-mik1i").version(21).model
        self.lane_model   = rf.workspace().project("two-wheeler-lane-detection").version(3).model

        logger.info("Initializing Seatbelt model...")
        self._seatbelt_ready = False
        if os.path.exists(SEATBELT_DETECTOR_PATH) and os.path.exists(SEATBELT_PREDICTOR_PATH):
            self.seatbelt_detector  = torch.hub.load("ultralytics/yolov5", "custom",
                                                      path=SEATBELT_DETECTOR_PATH, force_reload=False)
            self.seatbelt_predictor = load_model(SEATBELT_PREDICTOR_PATH, compile=False)
            self._seatbelt_ready = True
            logger.info("Seatbelt model loaded.")
        else:
            logger.warning(f"Seatbelt model files not found at {SEATBELT_DETECTOR_PATH} / "
                           f"{SEATBELT_PREDICTOR_PATH}. Seatbelt detection will be skipped.")

        logger.info("✅ All models initialized successfully.")

    # ── ANPR ──────────────────────────────────
    def run_anpr(self, image_path: str) -> dict:
        """Returns best plate text + confidence from image."""
        results = self.alpr.predict(image_path)
        if not results:
            return {"plate": None, "confidence": 0.0}
        # Pick highest confidence plate
        best = max(results, key=lambda r: r.ocr.confidence if r.ocr else 0)
        if best.ocr and best.ocr.text:
            conf = best.ocr.confidence
            if isinstance(conf, list):
                conf = sum(conf) / len(conf)
            return {"plate": best.ocr.text.upper(), "confidence": round(float(conf), 3)}
        return {"plate": None, "confidence": 0.0}

    # ── TRAFFIC VIOLATIONS ────────────────────
    def run_traffic_detection(self, image_path: str) -> dict:
        """Detects helmet, triple riding, wrong lane violations."""
        violations = []
        pred1 = self.helmet_model.predict(image_path, confidence=40, overlap=40).json()["predictions"]

        for pr in pred1:
            if pr["class"] != "motorcyclist":
                continue

            mx1 = int(pr["x"] - pr["width"] / 2)
            my1 = int(pr["y"] - pr["height"] / 2)
            mx2 = int(pr["x"] + pr["width"] / 2)
            my2 = int(pr["y"] + pr["height"] / 2)

            # Crop motorcyclist
            img = cv2.imread(image_path)
            crop = img[my1:my2, mx1:mx2]
            crop_path = "temp_moto_crop.jpg"
            cv2.imwrite(crop_path, crop)

            # Lane check
            lane_preds = self.lane_model.predict(crop_path, confidence=10, overlap=10).json()["predictions"]
            rear_detected = any(
                lp["class"] == "rear" and mx1 < lp["x"] < mx2 and my1 < lp["y"] < my2
                for lp in lane_preds
            )
            if rear_detected:
                violations.append("wrong_lane")

            # Face detection
            face_preds = self.face_model.predict(crop_path, confidence=40, overlap=30).json()["predictions"]
            num_faces = sum(
                1 for fp in face_preds
                if fp["class"] == "face" and mx1 < fp["x"] < mx2 and my1 < fp["y"] < my2
            )

            # Helmet detection
            helmet_preds = [p for p in pred1 if p["class"] == "helmet"
                            and mx1 < p["x"] < mx2 and my1 < p["y"] < my2]
            num_helmets = len(helmet_preds)
            helmet_detected = num_helmets > 0

            if not helmet_detected or num_faces > 1:
                violations.append("no_helmet")

            if num_faces + num_helmets > 2:
                violations.append("triple_riding")

            if os.path.exists(crop_path):
                os.remove(crop_path)

        return {"violations": list(set(violations))}

    # ── SEATBELT ──────────────────────────────
    def run_seatbelt_detection(self, image_path: str) -> dict:
        """Detects seatbelt violation."""
        if not self._seatbelt_ready:
            return {"violations": []}

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.seatbelt_detector(img_rgb)
        boxes = results.xyxy[0].cpu().numpy()

        violations = []
        for box in boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            crop = img_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (224, 224))
            crop_norm = (crop_resized / 127.5) - 1
            crop_exp = tf.expand_dims(crop_norm, axis=0)
            pred = self.seatbelt_predictor.predict(crop_exp)
            index = np.argmax(pred)
            score = pred[0][index]
            if index == 0 and score >= SEATBELT_THRESHOLD:  # No seatbelt
                violations.append("no_seatbelt")

        return {"violations": violations}

    # ── FULL PIPELINE ─────────────────────────
    def process_image(self, image_path: str) -> dict:
        """
        Run full pipeline on a single image.
        Returns: plate, all violations, total fine.
        """
        logger.info(f"Processing image: {image_path}")

        # Step 1: ANPR
        anpr_result = self.run_anpr(image_path)

        # Step 2: Traffic violations
        traffic_result = self.run_traffic_detection(image_path)

        # Step 3: Seatbelt
        seatbelt_result = self.run_seatbelt_detection(image_path)

        # Aggregate violations
        all_violations = list(set(traffic_result["violations"] + seatbelt_result["violations"]))

        # Calculate fines
        fine_breakdown = {v: FINE_AMOUNTS.get(v, 500) for v in all_violations}
        total_fine = sum(fine_breakdown.values())

        result = {
            "plate_number":    anpr_result["plate"],
            "plate_confidence": anpr_result["confidence"],
            "violations":      all_violations,
            "fine_breakdown":  fine_breakdown,
            "total_fine":      total_fine,
            "image_path":      image_path,
            "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        logger.info(f"Result: Plate={result['plate_number']}, "
                    f"Violations={all_violations}, Fine=₹{total_fine}")
        return result
