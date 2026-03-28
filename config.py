import os

# ─────────────────────────────────────────────
# BASE PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR       = os.path.join(BASE_DIR, "models")
CHALLANS_DIR     = os.path.join(BASE_DIR, "challans")
UPLOADS_DIR      = os.path.join(BASE_DIR, "static", "uploads")
DATABASE_PATH    = os.path.join(BASE_DIR, "database", "traffic.db")

# ─────────────────────────────────────────────
# MODEL PATHS
# ─────────────────────────────────────────────
HELMET_MODEL_PATH   = os.path.join(MODELS_DIR, "helmet.pt")
SEATBELT_MODEL_PATH = os.path.join(MODELS_DIR, "seatbelt.pt")
KERAS_MODEL_PATH    = os.path.join(MODELS_DIR, "keras_model.h5")

# ─────────────────────────────────────────────
# MODEL SETTINGS
# ─────────────────────────────────────────────
HELMET_CONF_THRESHOLD   = 0.40
SEATBELT_CONF_THRESHOLD = 0.40
ANPR_CONF_THRESHOLD     = 0.40

# ─────────────────────────────────────────────
# FINE AMOUNTS (in INR)
# ─────────────────────────────────────────────
FINES = {
    "no_helmet"     : 1000,
    "triple_riding" : 1000,
    "no_seatbelt"   : 1000,
    "wrong_lane"    : 500,
}

# ─────────────────────────────────────────────
# EMAIL SETTINGS (Gmail SMTP)
# ─────────────────────────────────────────────
EMAIL_HOST          = "smtp.gmail.com"
EMAIL_PORT          = 587
EMAIL_USE_TLS       = True
EMAIL_SENDER        = "sharavanrockzz20@gmail.com"       # ← change this
EMAIL_PASSWORD      = "yctvnncuytshpgxr"     # ← change this (Gmail App Password)
EMAIL_SUBJECT       = "Traffic Violation Challan"

# ─────────────────────────────────────────────
# FLASK SETTINGS
# ─────────────────────────────────────────────
SECRET_KEY          = "traffic_challan_secret_key_2024"
DEBUG               = True
HOST                = "0.0.0.0"
PORT                = 5000
MAX_CONTENT_LENGTH  = 16 * 1024 * 1024   # 16MB max upload

ALLOWED_EXTENSIONS  = {"png", "jpg", "jpeg", "mp4", "avi", "mov"}
