from ultralytics import YOLO
from config import HELMET_MODEL_PATH, HELMET_CONF_THRESHOLD
import os


class HelmetModel:
    def __init__(self):
        if not os.path.exists(HELMET_MODEL_PATH):
            print(f"WARNING: Helmet model not found at {HELMET_MODEL_PATH}")
            self.model = None
        else:
            self.model = YOLO(HELMET_MODEL_PATH)
            print(f"Helmet model loaded from {HELMET_MODEL_PATH}")

    def predict(self, frame) -> list:
        """
        Run helmet/triple riding detection on a frame.
        Returns list of dicts with class, confidence, bbox.
        """
        if self.model is None:
            return []

        results = self.model.predict(
            source=frame,
            conf=HELMET_CONF_THRESHOLD,
            verbose=False
        )

        detections = []
        for result in results:
            for box in result.boxes:
                class_id   = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "class"     : class_name,
                    "confidence": confidence,
                    "bbox"      : [x1, y1, x2, y2]
                })

        return detections
