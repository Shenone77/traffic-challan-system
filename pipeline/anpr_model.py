"""
pipeline/anpr_model.py  ─ FINAL WORKING VERSION
=================================================
Integrates the EXACT pipeline from:
  AI-based-indian-license-plate-detection-master

Flow:
  Full image
      ↓
  anpr_detector.pt (YOLOv11) → plate bbox → crop
      ↓
  _deskew() + _upscale()     → fix skewed YOLO crops
      ↓
  METHOD 1 ─ Haar cascade (indian_license_plate.xml) → plate region
           → segment_characters() → CNN classify each char
      ↓  (fallback if CNN gives < 6 chars)
  METHOD 2 ─ EasyOCR with beamsearch + CLAHE preprocessing
      ↓
  correct_plate_format() → positional char correction
  _best_candidate()      → regex-validated winner

CLASS ORDER (from data/train, sorted alphabetically by Keras):
  class_0..class_9 → index 0-9
  class_A..class_Z → index 10-35
  '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
"""

import os, re, cv2, numpy as np
from collections import Counter

# ── paths ─────────────────────────────────────────────────────────────────────
_DIR        = os.path.dirname(os.path.abspath(__file__))
_MODELS     = os.path.abspath(os.path.join(_DIR, '..', 'models'))

PLATE_DET   = os.path.join(_MODELS, 'anpr_detector.pt')
CNN_PATH    = os.path.join(_MODELS, 'best_model.pth')   # your trained Keras model
HAAR_XML    = os.path.join(_MODELS, 'indian_license_plate.xml')

# ── character index map (Keras ImageDataGenerator alphabetical sort) ──────────
# class_0→0, class_1→1 … class_9→9, class_A→10 … class_Z→35
CHARS   = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
IDX2CH  = {i: c for i, c in enumerate(CHARS)}

# ── position-aware correction maps ───────────────────────────────────────────
NUM2ALPHA = {'0':'O','1':'I','2':'Z','4':'A','5':'S','6':'G','8':'B'}
ALPHA2NUM = {'O':'0','Q':'0','D':'0','I':'1','L':'1','J':'1',
             'Z':'2','S':'5','B':'8','G':'6','A':'4'}

PLATE_RE = [
    re.compile(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'),   # MH12AB1234
    re.compile(r'^[A-Z]{2}\d{2}[A-Z]\d{4}$'),        # MH12A1234
    re.compile(r'^\d{2}BH\d{4}[A-Z]{1,2}$'),         # 22BH1234AB
    re.compile(r'^[A-Z]{2}\d{2}[A-Z]{3}\d{4}$'),     # 3-letter series
]

# All valid Indian state/UT codes — used for scoring
INDIAN_STATE_CODES = {
    "AP","AR","AS","BR","CG","DL","GA","GJ","HR","HP",
    "JH","JK","KA","KL","LA","LD","MH","ML","MN","MP",
    "MZ","NL","OD","PB","PY","RJ","SK","TN","TR","TS",
    "UK","UP","WB","AN","CH","DN","DD","HP","MG","KL",
}

# ── lazy singletons ───────────────────────────────────────────────────────────
_plate_model = _cnn = _reader = _haar = None


# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_plate_model():
    global _plate_model
    if _plate_model is None:
        from ultralytics import YOLO
        if not os.path.exists(PLATE_DET):
            print(f"[ANPR] anpr_detector.pt not found at {PLATE_DET}")
            return None
        _plate_model = YOLO(PLATE_DET)
        print("[ANPR] anpr_detector.pt loaded")
    return _plate_model


def _load_haar():
    """Haar cascade from the zip — more robust plate localiser for the crop."""
    global _haar
    if _haar is None:
        if os.path.exists(HAAR_XML):
            _haar = cv2.CascadeClassifier(HAAR_XML)
            print("[ANPR] Haar cascade loaded")
        else:
            print(f"[ANPR] Haar XML not found at {HAAR_XML} — skipping")
            _haar = False   # sentinel: don't try again
    return _haar if _haar is not False else None



# ── PyTorch MobileNetV2 wrapper ───────────────────────────────────────────────
# best_model.pth is a PyTorch MobileNetV2 state_dict
# (keys: features.0.0.weight, features.0.1.weight ... classifier.1.weight)
# We rebuild the exact MobileNetV2 architecture, load the weights,
# and wrap it in a predict() compatible interface.

class _TorchCNN:
    """
    Wraps a PyTorch MobileNetV2 char classifier so the rest of the
    code can call  model.predict(batch)  exactly like a Keras model.
    Output shape: (N, 36)  — softmax probabilities over 36 char classes.
    """
    def __init__(self, state_dict, num_classes=36):
        import torch
        import torch.nn as nn
        from torchvision.models import mobilenet_v2

        self._torch = torch

        # EXACT classifier head from best_model.pth:
        # classifier.1: Linear(1280 -> 256)
        # classifier.4: Linear(256  -> 36)
        net = mobilenet_v2(weights=None)
        net.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(net.last_channel, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(256, num_classes),
        )
        net.load_state_dict(state_dict, strict=True)
        net.eval()
        self.net = net

    def predict(self, x_np, verbose=0):
        """
        x_np: numpy array shape (N, 28, 28, 3), float32, values 0-1
        returns: numpy array shape (N, 36)
        """
        import torch
        import torch.nn.functional as F

        # MobileNetV2 expects (N, 3, H, W) and min 32x32 — resize to 32x32
        x = x_np.transpose(0, 3, 1, 2)           # NHWC → NCHW
        t = torch.from_numpy(x).float()
        t = F.interpolate(t, size=(32, 32), mode='bilinear', align_corners=False)

        # ImageNet normalisation (what MobileNetV2 was pretrained on)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        t    = (t - mean) / std

        with torch.no_grad():
            logits = self.net(t)
            probs  = torch.softmax(logits, dim=1)
        return probs.numpy()



def _load_reader():
    global _reader
    if _reader is None:
        import easyocr
        try:
            import torch
            gpu = torch.cuda.is_available()
        except:
            gpu = False
        _reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
        print(f'[ANPR] EasyOCR ready (GPU={gpu})')
    return _reader

def _build_cnn_arch():
    """Fallback Keras arch (only used if PyTorch load fails)."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
    m = Sequential([
        Conv2D(32, (24,24), input_shape=(28,28,3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.4),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(36,  activation='softmax'),
    ])
    return m


def _load_cnn():
    """
    Load best_model.pth.
    Strategy 1: PyTorch MobileNetV2 state_dict  ← what your file actually is
    Strategy 2: keras load_model
    Strategy 3: keras load_weights on rebuilt arch
    """
    global _cnn
    if _cnn is not None:
        return _cnn

    if not os.path.exists(CNN_PATH):
        print(f"[ANPR] best_model.pth not found — EasyOCR only")
        _cnn = False
        return None

    # ── Strategy 1: PyTorch MobileNetV2 (your actual model) ──────────────────
    try:
        import torch
        # check torchvision is available
        import torchvision
        state = torch.load(CNN_PATH, map_location='cpu')
        if isinstance(state, dict) and any('features' in k for k in state.keys()):
            # Determine num_classes from final classifier weight
            cls_key = [k for k in state.keys() if 'classifier' in k and 'weight' in k]
            num_classes = state[cls_key[-1]].shape[0] if cls_key else 36
            print(f"[ANPR] Detected MobileNetV2 state_dict, classes={num_classes}")
            wrapper = _TorchCNN(state, num_classes=num_classes)
            # quick smoke test
            dummy = np.zeros((1,28,28,3), dtype=np.float32)
            out   = wrapper.predict(dummy)
            assert out.shape[-1] == num_classes
            _cnn = wrapper
            print(f"[ANPR] MobileNetV2 loaded via PyTorch ✓  (classes={num_classes})")
            return _cnn
    except ImportError:
        print("[ANPR] torchvision not installed — run: pip install torchvision")
    except Exception as e:
        print(f"[ANPR] Strategy 1 (MobileNetV2 PyTorch): {e}")

    # ── Strategy 2: keras load_model directly ────────────────────────────────
    try:
        from tensorflow.keras.models import load_model
        m   = load_model(CNN_PATH, compile=False)
        dummy = np.zeros((1,28,28,3), dtype=np.float32)
        out   = m.predict(dummy, verbose=0)
        assert out.shape[-1] == 36
        _cnn = m
        print("[ANPR] CNN loaded via keras load_model ✓")
        return _cnn
    except Exception as e:
        print(f"[ANPR] Strategy 2 (keras load_model): {e}")

    # ── Strategy 3: keras load_weights on rebuilt arch ────────────────────────
    try:
        m = _build_cnn_arch()
        m.load_weights(CNN_PATH)
        dummy = np.zeros((1,28,28,3), dtype=np.float32)
        out   = m.predict(dummy, verbose=0)
        assert out.shape[-1] == 36
        _cnn = m
        print("[ANPR] CNN loaded via keras load_weights ✓")
        return _cnn
    except Exception as e:
        print(f"[ANPR] Strategy 3 (keras load_weights): {e}")

    print("[ANPR] WARNING: All CNN load strategies failed — EasyOCR only mode")
    _cnn = False
    return None

    global _reader
    if _reader is None:
        import easyocr, torch
        gpu = torch.cuda.is_available() if True else False
        try: gpu = torch.cuda.is_available()
        except: gpu = False
        _reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
        print(f"[ANPR] EasyOCR ready (GPU={gpu})")
    return _reader


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING  (fixes skewed YOLO crops — was completely missing before)
# ══════════════════════════════════════════════════════════════════════════════

def _upscale(img, min_h=80):
    h, w = img.shape[:2]
    if h < min_h:
        s = min_h / h
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_CUBIC)
    return img


def _order_pts(pts):
    rect = np.zeros((4,2), dtype=np.float32)
    s    = pts.sum(axis=1);    diff = np.diff(pts, axis=1)
    rect[0]=pts[np.argmin(s)]; rect[2]=pts[np.argmax(s)]
    rect[1]=pts[np.argmin(diff)]; rect[3]=pts[np.argmax(diff)]
    return rect


def _deskew(img):
    """Perspective-correct a skewed plate crop using contour quad detection."""
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 30, 200)
    cnts, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts    = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    for c in cnts:
        peri  = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018*peri, True)
        if len(approx) == 4:
            pts  = approx.reshape(4,2).astype(np.float32)
            rect = _order_pts(pts)
            tl,tr,br,bl = rect
            W = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
            H = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
            if W < 10 or H < 5: continue
            dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
            M   = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(img, M, (W, H))
    return img


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1 — CNN pipeline (exact from the zip)
# ══════════════════════════════════════════════════════════════════════════════

def _find_contours(dimensions, img):
    """
    Exact function from plate_detection.py / notebook cell 1.
    Returns array of (44,24) char patches sorted left→right.
    """
    lower_w, upper_w, lower_h, upper_h = dimensions
    cnts, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts    = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]

    x_list, patches = [], []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if lower_w < w < upper_w and lower_h < h < upper_h:
            x_list.append(x)
            patch = np.zeros((44, 24))
            ch    = img[y:y+h, x:x+w]
            ch    = cv2.resize(ch, (20, 40))
            ch    = cv2.subtract(255, ch)        # invert
            patch[2:42, 2:22] = ch
            patches.append(patch)

    if not patches:
        return np.array([])

    order   = sorted(range(len(x_list)), key=lambda k: x_list[k])
    return np.array([patches[i] for i in order])


def _segment_characters(plate_bgr):
    """
    Exact function from character_segmentation.py.
    Resize to 333×75, binarise, erode, dilate, find char contours.
    """
    img       = cv2.resize(plate_bgr, (333, 75))
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary    = cv2.erode(binary,  np.ones((3,3), np.uint8))
    binary    = cv2.dilate(binary, np.ones((3,3), np.uint8))

    H, W = binary.shape   # 75, 333
    binary[0:3,  :]   = 255
    binary[:,  0:3]   = 255
    binary[72:75, :]  = 255
    binary[:, 330:333] = 255

    # dimensions from the notebook: [LP_WIDTH/6, LP_WIDTH/2, LP_HEIGHT/10, 2*LP_HEIGHT/3]
    # NOTE: notebook has LP_WIDTH = shape[0] = 75 (height), LP_HEIGHT = shape[1] = 333 (width)
    dims = [H/6, H/2, W/10, 2*W/3]
    return _find_contours(dims, binary)


def _fix_dimension(img):
    """Convert (44,24) or (28,28) single-channel to (28,28,3) for the CNN."""
    img28 = cv2.resize(img.astype(np.float32), (28, 28))
    out   = np.zeros((28, 28, 3), dtype=np.float32)
    for i in range(3):
        out[:,:,i] = img28
    return out


def _cnn_predict_char(model, patch):
    """Predict one character patch → (char, confidence)."""
    inp  = _fix_dimension(patch).reshape(1, 28, 28, 3) / 255.0
    pred = model.predict(inp, verbose=0)[0]
    idx  = int(np.argmax(pred))
    return IDX2CH.get(idx, '?'), float(pred[idx])


def read_with_cnn(plate_bgr):
    """
    METHOD 1: segment chars → CNN classify → correct format.
    Tries both the raw crop and a Haar-localised sub-crop.
    """
    model = _load_cnn()
    if not model:
        return ''

    best_result = ''

    # try with and without Haar re-localisation inside the crop
    crops_to_try = [plate_bgr]

    haar = _load_haar()
    if haar is not None:
        gray_crop = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        rects = haar.detectMultiScale(gray_crop, scaleFactor=1.3, minNeighbors=5)
        if len(rects):
            x,y,w,h = rects[0]
            a = int(0.02*plate_bgr.shape[0])
            b = int(0.025*plate_bgr.shape[1])
            sub = plate_bgr[max(0,y+a):y+h-a, max(0,x+b):x+w-b]
            if sub.size > 0:
                crops_to_try.insert(0, sub)   # try Haar crop first

    for crop in crops_to_try:
        patches = _segment_characters(crop)
        if len(patches) < 6:
            # try after upscaling
            big     = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            patches = _segment_characters(big)

        if len(patches) < 6:
            print(f"[ANPR] CNN: only {len(patches)} chars segmented")
            continue

        chars   = [_cnn_predict_char(model, p) for p in patches]
        raw     = ''.join(c for c,_ in chars)
        avg_conf= sum(conf for _,conf in chars) / len(chars)
        fixed   = correct_plate_format(raw)
        print(f"[ANPR] CNN raw={raw}  fixed={fixed}  conf={avg_conf:.2f}")

        if fixed:
            if _validate(fixed):
                return fixed   # validated → done immediately
            if not best_result:
                best_result = fixed

    return best_result


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2 — EasyOCR fallback (with proper preprocessing)
# ══════════════════════════════════════════════════════════════════════════════

def read_with_easyocr(plate_bgr):
    reader = _load_reader()
    if reader is None:
        return ''

    h, w = plate_bgr.shape[:2]
    candidates = []

    for scale in [4, 3]:   # reduced from 3 scales to 2 for speed
        up  = cv2.resize(plate_bgr, (w*scale, h*scale),
                         interpolation=cv2.INTER_LANCZOS4)
        # sharpen
        up  = cv2.filter2D(up, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))

        # CLAHE binary variant
        gray   = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray   = clahe.apply(gray)
        _, bn  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        bn_bgr = cv2.cvtColor(bn, cv2.COLOR_GRAY2BGR)

        for variant in [up]:   # removed binary variant — speeds up 2x
            try:
                hits = reader.readtext(
                    variant, detail=1,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    paragraph=False, min_size=10,
                    text_threshold=0.25, low_text=0.25,
                    width_ths=0.9,           # merge fragmented chars
                    decoder='beamsearch',    # better sequence decoding
                    beamWidth=10,
                )
                for (_, text, conf) in hits:
                    if conf > 0.15 and len(text) >= 4:
                        fixed = correct_plate_format(text)
                        if fixed:
                            candidates.append((fixed, conf))
                            print(f"[ANPR] EasyOCR {scale}x → {text} → {fixed} ({conf:.2f})")

            except Exception as e:
                print(f"[ANPR] EasyOCR error: {e}")

    return _best_candidate_scored(candidates)


# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

# Visual similarity map for state code correction
_VISUAL_SIMILAR = {
    'T': ['M','I','L','F'],  'M': ['T','N','W'],
    'O': ['Q','C','G','D'],  'Q': ['O','C','G'],
    'Z': ['2','S','7'],      'I': ['1','L','J','P'],
    '0': ['O','Q','D'],      'N': ['M','H'],
    '1': ['I','L','J'],      'L': ['I','1'],
    'W': ['M','N','A'],      'A': ['W','H','R'],
    'P': ['I','F','R'],      'R': ['P','B','K'],
    'H': ['N','M','K'],      'V': ['U','Y'],
    'U': ['V','Y'],          'Y': ['V','U'],
    'B': ['R','P','8'],      'S': ['5','Z'],
    '5': ['S','Z'],          '8': ['B','3'],
}

def _snap_state_code(sc: str) -> str:
    """
    Snap misread state code to nearest valid Indian state code.
    Tries single char substitution first, then both chars together.
    e.g. TP→MP, WI→AP, OZ→OD
    """
    if sc in INDIAN_STATE_CODES:
        return sc
    # try replacing one char at a time
    for i in range(2):
        for alt in _VISUAL_SIMILAR.get(sc[i], []):
            candidate = sc[:i] + alt + sc[i+1:]
            if candidate in INDIAN_STATE_CODES:
                return candidate
    # try replacing both chars simultaneously
    for alt0 in _VISUAL_SIMILAR.get(sc[0], []):
        for alt1 in _VISUAL_SIMILAR.get(sc[1], []):
            candidate = alt0 + alt1
            if candidate in INDIAN_STATE_CODES:
                return candidate
    return sc


def correct_plate_format(raw):
    """
    Enforce Indian plate structure AA-00-AA-0000.
    pos 0,1 → letters   pos 2,3 → digits
    pos 4,5 → letters   pos 6-9 → digits
    Also snaps state code to nearest valid Indian state if misread.
    """
    text = re.sub(r'[^A-Z0-9]', '', raw.upper())
    # Only accept 9 or 10 char reads — anything shorter is too ambiguous
    # 9 chars = single letter series e.g. MH12A1234
    # 10 chars = standard e.g. MP04ZD7116
    if len(text) < 9 or len(text) > 10:
        return ''

    ch = list(text)
    for i in [0,1]:            # must be alpha
        if ch[i].isdigit(): ch[i] = NUM2ALPHA.get(ch[i], ch[i])
    for i in [2,3]:            # must be digit
        if ch[i].isalpha(): ch[i] = ALPHA2NUM.get(ch[i], '0')
    for i in [4,5]:            # must be alpha
        if ch[i].isdigit(): ch[i] = NUM2ALPHA.get(ch[i], ch[i])
    for i in [6,7,8,9]:        # must be digit
        if ch[i].isalpha(): ch[i] = ALPHA2NUM.get(ch[i], '0')

    # snap state code to nearest valid Indian state
    state = ''.join(ch[:2])
    snapped = _snap_state_code(state)
    if snapped != state:
        print(f'[ANPR] state snap: {state} → {snapped}')
        ch[0], ch[1] = snapped[0], snapped[1]

    return ''.join(ch)


def _validate(text):
    return any(p.match(text) for p in PLATE_RE)


def _has_valid_state_code(text: str) -> bool:
    return len(text) >= 2 and text[:2].upper() in INDIAN_STATE_CODES

def _best_candidate_scored(candidates):
    """
    Three-tier scoring:
      Tier 1: valid regex + valid Indian state code  (best)
      Tier 2: valid regex only
      Tier 3: no validation (worst)
    Within each tier: confidence × frequency bonus.
    """
    if not candidates:
        return ""

    def tier(text):
        if _validate(text) and _has_valid_state_code(text):
            return 3   # best
        if _validate(text):
            return 2
        return 1

    freq = Counter(t for t, _ in candidates)
    scored = []
    for text, conf in candidates:
        t = tier(text)
        score = t * 10 + conf * (1 + 0.3 * freq[text])
        scored.append((text, score))
        print(f"[ANPR] scoring {text}: tier={t} conf={conf:.2f} score={score:.2f}")

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


# ══════════════════════════════════════════════════════════════════════════════
# ANPRModel — drop-in for pipeline/orchestrator.py and video_pipeline.py
# ══════════════════════════════════════════════════════════════════════════════

class ANPRModel:
    """
    Usage (unchanged from original):
        results = anpr_model.predict(frame)
        best    = max(results, key=lambda x: x["confidence"])
        plate   = best["plate_text"]
        x1,y1,x2,y2 = best["bbox"]
    """

    def __init__(self):
        _load_plate_model()
        _load_cnn()
        _load_haar()
        _load_reader()

    def predict(self, image_bgr, conf_threshold=0.30):
        model = _load_plate_model()
        if model is None or image_bgr is None:
            return []

        dets = model.predict(source=image_bgr, conf=conf_threshold,
                             verbose=False, imgsz=640)
        results = []

        for det in dets:
            for box in det.boxes:
                conf         = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                ih, iw       = image_bgr.shape[:2]

                # trim 5% edges to remove vehicle body noise
                bw, bh = x2-x1, y2-y1
                tx, ty = int(bw*0.05), int(bh*0.05)
                x1c = max(0,  x1+tx);  y1c = max(0,  y1+ty)
                x2c = min(iw, x2-tx);  y2c = min(ih, y2-ty)
                raw_crop   = image_bgr[y1c:y2c, x1c:x2c]

                # deskew + upscale before OCR
                plate_crop = _deskew(_upscale(raw_crop))

                # EasyOCR with CLAHE + beamsearch (primary)
                plate_text = read_with_easyocr(plate_crop)

                # fallback: EasyOCR on raw un-deskewed crop
                if not plate_text:
                    plate_text = read_with_easyocr(raw_crop)

                results.append({
                    "plate_text": plate_text or "UNKNOWN",
                    "confidence": conf,
                    "bbox":       (x1, y1, x2, y2),
                    "plate_crop": plate_crop,
                })

        return results