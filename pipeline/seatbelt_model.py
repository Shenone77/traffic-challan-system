import cv2
import numpy as np
import os
from ultralytics import YOLO
from config import SEATBELT_MODEL_PATH, SEATBELT_CONF_THRESHOLD


class SeatbeltModel:
    def __init__(self):
        if not os.path.exists(SEATBELT_MODEL_PATH):
            print(f"WARNING: Seatbelt model not found at {SEATBELT_MODEL_PATH}")
            self.model = None
        else:
            self.model = YOLO(SEATBELT_MODEL_PATH)
            print(f"Seatbelt model loaded from {SEATBELT_MODEL_PATH}")
            # Log actual class names so we can verify
            print(f"[Seatbelt] model.names = {self.model.names}")

        # Hardcoded fallback — these match how the model was trained
        self.class_names = {0: "seatbelt_worn", 1: "no_seatbelt"}

    def predict(self, frame) -> list:
        if self.model is None:
            return []

        results = self.model.predict(
            source  = frame,
            conf    = SEATBELT_CONF_THRESHOLD,
            verbose = False
        )

        detections = []
        for result in results:
            for box in result.boxes:
                class_id   = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Use hardcoded names — guaranteed to match violation check strings
                class_name = self.class_names.get(class_id, self.model.names.get(class_id, f"class_{class_id}"))

                detections.append({
                    "class"     : class_name,
                    "confidence": confidence,
                    "bbox"      : [x1, y1, x2, y2]
                })
                print(f"[Seatbelt] detected: {class_name} conf={confidence:.2f}")

        return detections