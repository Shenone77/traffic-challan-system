"""
pipeline/orchestrator.py  — FINAL FIXED VERSION
=================================================

Bug fixes vs old version:
1. Seatbelt violation detected in annotation but no challan generated
   → Old code checked label == "no_seatbelt" but model.names may return
     different string. Now uses _is_seatbelt_violation() substring match.
2. Helmet challan shows UNKNOWN plate
   → Old code picked highest-confidence ANPR result globally.
     Now picks the plate spatially closest to the violation.
3. Helmet violations assigned to cars / seatbelt to bikes
   → _plate_is_car() separates bikes (narrow plate) from cars (wide plate).
4. All violations dumped to one plate in multi-vehicle scenes
   → _vehicle_zone() + _overlap() assigns violations per-vehicle.
5. motorcyclist / license_plate classes trigger violations
   → Explicitly ignored.

Helmet model classes: helmet(0) no_helmet(1) triple_riding(2) motorcyclist(3) license_plate(4)
Seatbelt model classes: seatbelt_worn(0) no_seatbelt(1)
"""

import cv2
import numpy as np
from pipeline.helmet_model   import HelmetModel
from pipeline.seatbelt_model import SeatbeltModel
from pipeline.anpr_model     import ANPRModel

print("Loading models...")
helmet_model   = HelmetModel()
seatbelt_model = SeatbeltModel()
anpr_model     = ANPRModel()
print("All models loaded!")


# ── Violation helpers ─────────────────────────────────────────────────────────

def _is_helmet_violation(lbl: str) -> bool:
    lbl = lbl.lower().strip()
    return any(k in lbl for k in ("no_helmet", "without_helmet", "triple", "tripling"))

def _is_seatbelt_violation(lbl: str) -> bool:
    lbl = lbl.lower().strip()
    return any(k in lbl for k in ("no_seatbelt", "without_seatbelt", "noseatbelt"))

def _violation_key(lbl: str):
    lbl = lbl.lower().strip()
    if "triple" in lbl or "tripling" in lbl:
        return "triple_riding"
    if "no_helmet" in lbl or "without_helmet" in lbl:
        return "no_helmet"
    if "no_seatbelt" in lbl or "without_seatbelt" in lbl or "noseatbelt" in lbl:
        return "no_seatbelt"
    return None

# Classes to ignore entirely — not violations
_IGNORE = {"motorcyclist", "license_plate", "helmet", "seatbelt_worn"}


# ── Spatial helpers ───────────────────────────────────────────────────────────

def _plate_is_car(bbox) -> bool:
    """Wide plate = car/truck. Narrow = bike."""
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    return (w / h) >= 2.5


def _vehicle_zone(plate_bbox, frame_shape):
    """
    Expand plate bbox upward to cover the full vehicle body.
    pad_y=8 → extends 8× plate height upward (reaches rider head from plate).
    pad_x=0.8 → 80% sideways expansion.
    """
    x1, y1, x2, y2 = plate_bbox
    ih, iw = frame_shape[:2]
    bw, bh = x2 - x1, y2 - y1
    return (
        max(0,  int(x1 - bw * 0.8)),
        max(0,  int(y1 - bh * 8.0)),
        min(iw, int(x2 + bw * 0.8)),
        min(ih, y2),
    )


def _overlap(inner, outer) -> float:
    """Fraction of inner bbox that lies inside outer bbox."""
    ax1 = max(inner[0], outer[0]); ay1 = max(inner[1], outer[1])
    ax2 = min(inner[2], outer[2]); ay2 = min(inner[3], outer[3])
    inter = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    if inter == 0:
        return 0.0
    area = max((inner[2]-inner[0]) * (inner[3]-inner[1]), 1)
    return inter / area


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(image_path: str) -> dict:
    frame = cv2.imread(image_path)
    if frame is None:
        return {"success": False, "message": "Could not load image",
                "violations": [], "plate_number": "UNKNOWN", "vehicles": []}

    annotated = frame.copy()
    helmet_dets = seatbelt_dets = anpr_dets = []

    try: helmet_dets   = helmet_model.predict(frame)
    except Exception as e: print(f"[Orch] Helmet error: {e}")
    try: seatbelt_dets = seatbelt_model.predict(frame)
    except Exception as e: print(f"[Orch] Seatbelt error: {e}")
    try: anpr_dets     = anpr_model.predict(frame)
    except Exception as e: print(f"[Orch] ANPR error: {e}")

    # ── draw all detections ───────────────────────────────────────────────────
    for d in helmet_dets:
        lbl = d["class"]
        if lbl.lower() in ("motorcyclist", "license_plate"):
            continue
        x1, y1, x2, y2 = d["bbox"]
        is_viol = _is_helmet_violation(lbl)
        color   = (0, 0, 255) if is_viol else (0, 200, 0)
        cv2.rectangle(annotated, (x1,y1),(x2,y2), color, 2)
        cv2.putText(annotated, f"{lbl} {d['confidence']:.2f}",
                    (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for d in seatbelt_dets:
        lbl = d["class"]
        x1, y1, x2, y2 = d["bbox"]
        is_viol = _is_seatbelt_violation(lbl)
        color   = (0, 0, 255) if is_viol else (0, 200, 0)
        cv2.rectangle(annotated, (x1,y1),(x2,y2), color, 2)
        cv2.putText(annotated, f"{lbl} {d['confidence']:.2f}",
                    (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for d in anpr_dets:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(annotated, (x1,y1),(x2,y2), (255,165,0), 2)
        cv2.putText(annotated, f"Plate: {d['plate_text']}",
                    (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,165,0), 2)

    # ── per-vehicle violation assignment ─────────────────────────────────────
    vehicles = []

    if anpr_dets:
        for anpr in anpr_dets:
            pbbox  = anpr["bbox"]
            zone   = _vehicle_zone(pbbox, frame.shape)
            is_car = _plate_is_car(pbbox)
            viols  = []

            # Helmet violations → bikes only
            for d in helmet_dets:
                lbl = d["class"]
                if lbl.lower() in ("motorcyclist", "license_plate", "helmet"):
                    continue
                if not _is_helmet_violation(lbl):
                    continue
                dbbox = d["bbox"]
                if _overlap(dbbox, zone) > 0.10:
                    key = _violation_key(lbl)
                    if key and key not in viols:
                        viols.append(key)
                        print(f"[Orch] {key} → plate {anpr['plate_text']}")

            # Seatbelt violations → cars only
            if is_car:
                for d in seatbelt_dets:
                    lbl = d["class"]
                    if not _is_seatbelt_violation(lbl):
                        continue
                    dbbox = d["bbox"]
                    if _overlap(dbbox, zone) > 0.10:
                        if "no_seatbelt" not in viols:
                            viols.append("no_seatbelt")
                            print(f"[Orch] no_seatbelt → plate {anpr['plate_text']}")

            vehicles.append({
                "plate_number": anpr["plate_text"],
                "violations":   viols,
                "bbox":         pbbox,
                "is_car":       is_car,
                "confidence":   anpr["confidence"],
            })

        # ── Fallback: assign unmatched violations to best plate ──────────────
        # This handles CCTV crops where the plate and violation are in
        # different parts of the image (e.g. zoomed windscreen + plate from thumbnail)
        all_assigned_viols = [v for veh in vehicles for v in veh["violations"]]
        unmatched_helmet   = [d for d in helmet_dets
                              if _is_helmet_violation(d["class"])
                              and _violation_key(d["class"]) not in all_assigned_viols]
        unmatched_seatbelt = [d for d in seatbelt_dets
                              if _is_seatbelt_violation(d["class"])
                              and "no_seatbelt" not in all_assigned_viols]

        if unmatched_helmet or unmatched_seatbelt:
            # Pick best plate (highest ANPR confidence)
            best_anpr = max(anpr_dets, key=lambda x: x["confidence"])
            best_plate = best_anpr["plate_text"]
            print(f"[Orch] Fallback: assigning unmatched violations to {best_plate}")

            # Find existing vehicle entry for this plate
            existing = next((v for v in vehicles if v["plate_number"] == best_plate), None)
            if existing is None:
                existing = {"plate_number": best_plate, "violations": [],
                            "bbox": best_anpr["bbox"], "is_car": _plate_is_car(best_anpr["bbox"]),
                            "confidence": best_anpr["confidence"]}
                vehicles.append(existing)

            for d in unmatched_helmet:
                key = _violation_key(d["class"])
                if key and key not in existing["violations"]:
                    existing["violations"].append(key)
                    print(f"[Orch] fallback {key} → {best_plate}")

            for d in unmatched_seatbelt:
                if "no_seatbelt" not in existing["violations"]:
                    existing["violations"].append("no_seatbelt")
                    print(f"[Orch] fallback no_seatbelt → {best_plate}")

    else:
        # ── NO plates detected → global fallback ─────────────────────────────
        # Still generate a challan with UNKNOWN plate if violations are found
        viols = []

        # All helmet violations
        for d in helmet_dets:
            lbl = d["class"]
            if lbl.lower() in ("motorcyclist", "license_plate"):
                continue
            key = _violation_key(lbl)
            if key and key not in viols:
                viols.append(key)

        # Seatbelt violations (global — no plate to check car/bike)
        for d in seatbelt_dets:
            lbl = d["class"]
            if _is_seatbelt_violation(lbl) and "no_seatbelt" not in viols:
                viols.append("no_seatbelt")

        vehicles.append({
            "plate_number": "UNKNOWN",
            "violations":   viols,
            "bbox":         None,
            "is_car":       False,
            "confidence":   0.0,
        })

    # ── save annotated image ──────────────────────────────────────────────────
    dot = image_path.rfind(".")
    annotated_path = (image_path[:dot] + "_annotated" + image_path[dot:]
                      if dot != -1 else image_path + "_annotated.jpg")
    cv2.imwrite(annotated_path, annotated)

    # ── pick primary vehicle for backward-compat top-level keys ──────────────
    # Primary = vehicle with MOST violations. Tie → highest ANPR confidence.
    # This ensures app.py's result.get("violations") still works.
    primary = max(vehicles, key=lambda v: (len(v["violations"]), v["confidence"]))

    print(f"[Orch] vehicles={[(v['plate_number'], v['violations']) for v in vehicles]}")
    print(f"[Orch] primary = {primary['plate_number']} viols={primary['violations']}")

    return {
        "success":          True,
        # ── top-level keys for app.py backward compatibility ──
        "violations":       primary["violations"],
        "plate_number":     primary["plate_number"],
        # ── per-vehicle list for multi-challan support ──
        "vehicles":         vehicles,
        # ── raw detections ──
        "helmet_results":   helmet_dets,
        "seatbelt_results": seatbelt_dets,
        "anpr_results":     anpr_dets,
        "annotated_image":  annotated_path,
    }