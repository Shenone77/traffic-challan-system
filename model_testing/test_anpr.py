from ultralytics import YOLO
import cv2
import pytesseract
import os
import re

# -------------------------------
# 🔥 FORCE TESSERACT PATH (FIX)
# -------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------------
# PATHS
# -------------------------------
MODEL_PATH = "../models/anpr_detector.pt"
IMAGE_PATH = "../Test_images/Screenshot 2026-03-24 145335.png"

# -------------------------------
# LOAD MODEL
# -------------------------------
model = YOLO(MODEL_PATH)

# -------------------------------
# CHECK IMAGE
# -------------------------------
if not os.path.exists(IMAGE_PATH):
    print("❌ Image not found:", IMAGE_PATH)
    exit()

# -------------------------------
# READ IMAGE
# -------------------------------
img = cv2.imread(IMAGE_PATH)

# -------------------------------
# DETECT NUMBER PLATE
# -------------------------------
results = model(IMAGE_PATH)

for r in results:
    boxes = r.boxes.xyxy

    if boxes is None or len(boxes) == 0:
        print("❌ No plate detected")
        exit()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # Draw detection
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # -------------------------------
        # TIGHT CROP
        # -------------------------------
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.tolist())

        # Draw detection
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # -------------------------------
        # SIMPLE CROP (NO TIGHT CROPPING)
        # -------------------------------
            h, w = img.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            plate = img[y1:y2, x1:x2]

            if plate is None or plate.size == 0:
                continue

        # -------------------------------
        # 🔥 PREPROCESS FOR OCR
        # -------------------------------
        plate = cv2.resize(plate, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        if gray is None or gray.size == 0:
            continue

        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(gray, config=custom_config)
        # Adaptive threshold
        _,thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        # -------------------------------
        # 🔥 OCR (TESSERACT)
        # -------------------------------

        text = pytesseract.image_to_string(thresh, config=custom_config)

        print("🔍 Raw OCR:", text)

        # -------------------------------
        # CLEAN TEXT
        # -------------------------------
        text = text.strip()
        text = re.sub(r'[^A-Z0-9]', '', text)

        # -------------------------------
        # FIX COMMON ERRORS
        # -------------------------------
        def fix_common_errors(t):
            t = t.replace("O", "0")
            t = t.replace("I", "1")
            t = t.replace("Z", "2")
            return t

        text = fix_common_errors(text)

        # -------------------------------
        # FORMAT INDIAN PLATE
        # AA 00 AAAA 0000
        # -------------------------------
        def format_plate(t):
            pattern = r'^([A-Z]{2})(\d{2})([A-Z]{1,4})(\d{3,4})$'
            match = re.match(pattern, t)

            if match:
                return f"{match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}"

            return t
        final_text = format_plate(text)

        # -------------------------------
        # SHOW OUTPUT
        # -------------------------------
        cv2.imshow("Detected Plate", plate)
        cv2.imshow("Processed", thresh)
        cv2.imshow("Full Image", img)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break

        print("✅ FINAL PLATE:", final_text)

cv2.destroyAllWindows()