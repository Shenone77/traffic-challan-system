from flask import (Flask, render_template, request, jsonify,
                   send_file, redirect, url_for, flash, Response)
import os
from datetime import datetime
from werkzeug.utils import secure_filename

from config import *
from database import (
    init_db, add_owner, update_owner, get_owner, get_all_owners,
    delete_owner, create_challan, update_challan_pdf, update_challan_status,
    delete_challan, get_challan, get_all_challans, get_dashboard_stats
)
from utils.challan_generator import generate_challan_pdf
from utils.email_sender import send_challan_email

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

os.makedirs(UPLOADS_DIR,  exist_ok=True)
os.makedirs(CHALLANS_DIR, exist_ok=True)

IMAGE_EXT = {"jpg", "jpeg", "png", "webp", "bmp"}
VIDEO_EXT = {"mp4", "avi", "mov", "mkv", "wmv"}

def ext(filename):
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

def allowed(filename):
    return ext(filename) in IMAGE_EXT | VIDEO_EXT

def is_video(filename):
    return ext(filename) in VIDEO_EXT

def web_path(abs_path):
    """Convert absolute path → web URL starting from static/"""
    if not abs_path:
        return None
    p = abs_path.replace("\\", "/")
    i = p.find("static/")
    return p[i:] if i != -1 else p

def auto_email(challan_id, plate, violations, fine_details, total_fine,
               pdf_path, image_path=None):
    owner = get_owner(plate)
    if owner and owner.get("email"):
        r = send_challan_email(
            to_email=owner["email"], owner_name=owner["owner_name"],
            challan_id=challan_id, plate_number=plate,
            violations=violations, fine_details=fine_details,
            total_fine=total_fine, pdf_path=pdf_path)
        if r["success"]:
            update_challan_status(challan_id, "sent")
            return True, owner["email"]
        return False, r["message"]
    return False, "Owner not registered"


# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           stats=get_dashboard_stats(),
                           challans=get_all_challans()[:10])


# ── Detection ─────────────────────────────────────────────────────────────────
@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "GET":
        return render_template("detect.html")

    # ── save uploaded file ────────────────────────────────────────────────
    f = request.files.get("file")
    if not f or f.filename == "":
        return jsonify({"ok": False, "msg": "No file uploaded"})
    if not allowed(f.filename):
        return jsonify({"ok": False, "msg": "Unsupported file type"})

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{secure_filename(f.filename)}"
    fpath    = os.path.join(UPLOADS_DIR, filename)
    f.save(fpath)
    location = request.form.get("location", "Unknown")

    # ── run pipeline ──────────────────────────────────────────────────────
    try:
        if not is_video(filename):
            from pipeline.orchestrator import run_pipeline
            result = run_pipeline(fpath)
        else:
            from pipeline.video_pipeline import run_video_pipeline
            def _prog(pct, msg):
                _video_progress["pct"] = pct
                _video_progress["msg"] = msg
            result = run_video_pipeline(fpath, save_dir=UPLOADS_DIR,
                                        progress_cb=_prog)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "msg": str(e)})

    violations = result.get("violations", [])
    plate      = result.get("plate_number", "UNKNOWN")
    ann_img    = result.get("annotated_image")

    if not violations:
        return jsonify({"ok": True, "violations": [],
                        "plate": plate,
                        "annotated_image": web_path(ann_img),
                        "challan_id": None})

    # ── create challan ────────────────────────────────────────────────────
    fine_details = {v: FINES.get(v, 500) for v in violations}
    total_fine   = sum(fine_details.values())
    owner        = get_owner(plate)

    cid = create_challan(plate_number=plate, violations=violations,
                         fine_details=fine_details, total_fine=total_fine,
                         image_path=ann_img, location=location)

    pdf = generate_challan_pdf(challan_id=cid, plate_number=plate,
                               violations=violations, fine_details=fine_details,
                               total_fine=total_fine, owner=owner,
                               image_path=ann_img, annotated_image=ann_img)
    update_challan_pdf(cid, pdf)
    auto_email(cid, plate, violations, fine_details, total_fine, pdf, ann_img)

    return jsonify({"ok": True, "violations": violations,
                    "plate": plate, "total_fine": total_fine,
                    "fine_details": fine_details,
                    "challan_id": cid,
                    "annotated_image": web_path(ann_img),
                    "annotated_video": web_path(result.get("annotated_video"))})


# ── Live camera stream ────────────────────────────────────────────────────────
_live_running = False

_video_progress = {"pct": 0, "msg": "Processing..."}
_live_state     = {"violations": [], "plate": "UNKNOWN", "challan_id": None}

@app.route("/detect/progress")
def detect_progress():
    return jsonify(_video_progress)

@app.route("/live/detections")
def live_detections():
    return jsonify(_live_state)

@app.route("/live/feed")
def live_feed():
    global _live_running
    cam = int(request.args.get("camera", 0))
    _live_running = True
    from pipeline.video_pipeline import run_live_pipeline

    def gen():
        for jpeg_bytes, _ in run_live_pipeline(cam):
            if not _live_running:
                break
            yield jpeg_bytes

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/live/stop")
def live_stop():
    global _live_running
    _live_running = False
    return jsonify({"ok": True})


# ── Challans ──────────────────────────────────────────────────────────────────
@app.route("/challans")
def challans():
    return render_template("challans.html", challans=get_all_challans())

@app.route("/challan/<int:challan_id>")
def view_challan(challan_id):
    c = get_challan(challan_id)
    if not c:
        flash("Challan not found", "danger")
        return redirect(url_for("challans"))
    return render_template("view_challan.html",
                           challan=c, owner=get_owner(c["plate_number"]))

@app.route("/challan/<int:challan_id>/pdf")
def download_pdf(challan_id):
    c = get_challan(challan_id)
    if not c or not c.get("pdf_path"):
        flash("PDF not available", "danger")
        return redirect(url_for("challans"))
    return send_file(c["pdf_path"], as_attachment=True,
                     download_name=f"challan_{challan_id}.pdf")

@app.route("/challan/<int:challan_id>/send", methods=["POST"])
def send_challan_route(challan_id):
    c = get_challan(challan_id)
    if not c:
        return jsonify({"ok": False, "msg": "Not found"})
    owner = get_owner(c["plate_number"])
    if not owner or not owner.get("email"):
        return jsonify({"ok": False, "msg": "Owner email not registered"})
    r = send_challan_email(to_email=owner["email"], owner_name=owner["owner_name"],
                           challan_id=challan_id, plate_number=c["plate_number"],
                           violations=c["violations"], fine_details=c["fine_details"],
                           total_fine=c["total_fine"], pdf_path=c.get("pdf_path"))
    if r["success"]:
        update_challan_status(challan_id, "sent")
        return jsonify({"ok": True, "msg": f"Sent to {owner['email']}"})
    return jsonify({"ok": False, "msg": r["message"]})

@app.route("/challan/<int:challan_id>/status", methods=["POST"])
def challan_status(challan_id):
    s = request.json.get("status")
    if s not in ["pending", "sent", "paid"]:
        return jsonify({"ok": False, "msg": "Invalid status"})
    update_challan_status(challan_id, s)
    return jsonify({"ok": True})

@app.route("/challan/<int:challan_id>/delete", methods=["POST"])
def challan_delete(challan_id):
    c = get_challan(challan_id)
    if not c:
        return jsonify({"ok": False, "msg": "Not found"})
    delete_challan(challan_id)
    return jsonify({"ok": True, "msg": f"Challan #{challan_id} deleted"})


# ── Owners ────────────────────────────────────────────────────────────────────
@app.route("/owners")
def owners():
    return render_template("owners.html", owners=get_all_owners())

@app.route("/owners/add", methods=["POST"])
def owner_add():
    d     = request.form
    plate = d.get("plate_number", "").upper().strip()
    r     = add_owner(plate_number=plate, owner_name=d.get("owner_name",""),
                      phone=d.get("phone",""), email=d.get("email",""),
                      address=d.get("address",""),
                      vehicle_type=d.get("vehicle_type","unknown"))
    flash(r["message"], "success" if r["success"] else "danger")
    if r["success"]:
        for c in [c for c in get_all_challans()
                  if c["plate_number"]==plate and c["status"]=="pending"]:
            sent, info = auto_email(c["id"], plate, c["violations"],
                                    c["fine_details"], c["total_fine"],
                                    c.get("pdf_path"))
            if sent:
                flash(f"📧 Challan #{c['id']} auto-sent to {info}", "success")
    return redirect(url_for("owners"))

@app.route("/owners/edit/<plate>", methods=["POST"])
def owner_edit(plate):
    d = request.form
    update_owner(plate_number=plate, owner_name=d.get("owner_name",""),
                 phone=d.get("phone",""), email=d.get("email",""),
                 address=d.get("address",""),
                 vehicle_type=d.get("vehicle_type","unknown"))
    flash("Owner updated", "success")
    return redirect(url_for("owners"))

@app.route("/owners/delete/<plate>", methods=["POST"])
def owner_delete(plate):
    delete_owner(plate)
    flash("Owner deleted", "success")
    return redirect(url_for("owners"))

@app.route("/owners/search")
def owner_search():
    plate = request.args.get("plate","").upper().strip()
    o = get_owner(plate)
    return jsonify(o if o else {"error": "Not found"})

@app.route("/owners/upload-csv", methods=["POST"])
def owner_csv():
    from utils.csv_upload import process_csv
    f = request.files.get("csv_file")
    if not f or not f.filename.endswith(".csv"):
        flash("Please upload a .csv file", "danger")
        return redirect(url_for("owners"))
    res = process_csv(f)
    if res["success"]: flash(f"✅ {res['success']} owners added", "success")
    if res["failed"]:  flash(f"⚠️ {res['failed']} rows failed", "warning")
    return redirect(url_for("owners"))

@app.route("/owners/sample-csv")
def sample_csv():
    from utils.csv_upload import generate_sample_csv
    return Response(generate_sample_csv(), mimetype="text/csv",
                    headers={"Content-Disposition":
                             "attachment; filename=sample_owners.csv"})

@app.route("/api/stats")
def api_stats():
    return jsonify(get_dashboard_stats())


if __name__ == "__main__":
    init_db()
    print("🚦 Traffic Challan System")
    print(f"🌐 http://localhost:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)