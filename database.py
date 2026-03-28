import sqlite3
import json
import os
from datetime import datetime
from config import DATABASE_PATH


def get_connection():
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_owners (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT    UNIQUE NOT NULL,
            owner_name   TEXT    NOT NULL,
            phone        TEXT,
            email        TEXT,
            address      TEXT,
            vehicle_type TEXT    DEFAULT 'unknown',
            created_at   TEXT    DEFAULT (datetime('now','localtime')),
            updated_at   TEXT    DEFAULT (datetime('now','localtime'))
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS challans (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT    NOT NULL,
            violations   TEXT    NOT NULL,
            fine_details TEXT    NOT NULL,
            total_fine   INTEGER NOT NULL,
            status       TEXT    DEFAULT 'pending',
            image_path   TEXT,
            pdf_path     TEXT,
            location     TEXT    DEFAULT 'Unknown',
            created_at   TEXT    DEFAULT (datetime('now','localtime')),
            sent_at      TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS fines (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            violation_type TEXT    UNIQUE NOT NULL,
            fine_amount    INTEGER NOT NULL,
            description    TEXT
        )
    """)
    c.executemany("""
        INSERT OR IGNORE INTO fines (violation_type, fine_amount, description)
        VALUES (?, ?, ?)
    """, [
        ("no_helmet",     1000, "Riding without helmet"),
        ("triple_riding", 1000, "More than 2 riders on motorcycle"),
        ("no_seatbelt",   1000, "Driving without seatbelt"),
        ("wrong_lane",    500,  "Riding in wrong lane"),
    ])
    conn.commit()
    conn.close()
    print("Database initialized successfully!")


# ── Owners ────────────────────────────────────────────────────────────────────

def add_owner(plate_number, owner_name, phone, email, address="", vehicle_type="unknown"):
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO vehicle_owners
              (plate_number, owner_name, phone, email, address, vehicle_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (plate_number.upper().strip(), owner_name, phone, email, address, vehicle_type))
        conn.commit()
        return {"success": True, "message": f"Owner added for plate {plate_number}"}
    except sqlite3.IntegrityError:
        return {"success": False, "message": f"Plate {plate_number} already exists"}
    finally:
        conn.close()


def update_owner(plate_number, owner_name, phone, email, address="", vehicle_type="unknown"):
    conn = get_connection()
    conn.execute("""
        UPDATE vehicle_owners
        SET owner_name=?, phone=?, email=?, address=?, vehicle_type=?,
            updated_at=datetime('now','localtime')
        WHERE plate_number=?
    """, (owner_name, phone, email, address, vehicle_type,
          plate_number.upper().strip()))
    conn.commit()
    conn.close()
    return {"success": True, "message": "Owner updated successfully"}


def get_owner(plate_number):
    conn = get_connection()
    row  = conn.execute(
        "SELECT * FROM vehicle_owners WHERE plate_number=?",
        (plate_number.upper().strip(),)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_owners():
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM vehicle_owners ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_owner(plate_number):
    conn = get_connection()
    conn.execute("DELETE FROM vehicle_owners WHERE plate_number=?",
                 (plate_number.upper().strip(),))
    conn.commit()
    conn.close()
    return {"success": True, "message": "Owner deleted"}


# ── Challans ──────────────────────────────────────────────────────────────────

def create_challan(plate_number, violations, fine_details, total_fine,
                   image_path=None, location="Unknown"):
    conn = get_connection()
    cur  = conn.execute("""
        INSERT INTO challans
          (plate_number, violations, fine_details, total_fine, image_path, location)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (plate_number.upper().strip(), json.dumps(violations),
          json.dumps(fine_details), total_fine, image_path, location))
    cid = cur.lastrowid
    conn.commit()
    conn.close()
    return cid


def update_challan_pdf(challan_id, pdf_path):
    conn = get_connection()
    conn.execute("UPDATE challans SET pdf_path=? WHERE id=?", (pdf_path, challan_id))
    conn.commit()
    conn.close()


def update_challan_status(challan_id, status):
    conn   = get_connection()
    sent_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if status == "sent" else None
    conn.execute("UPDATE challans SET status=?, sent_at=? WHERE id=?",
                 (status, sent_at, challan_id))
    conn.commit()
    conn.close()


def delete_challan(challan_id):
    """Delete a challan record (used for paid/cleared challans)."""
    conn = get_connection()
    conn.execute("DELETE FROM challans WHERE id=?", (challan_id,))
    conn.commit()
    conn.close()
    return {"success": True, "message": f"Challan #{challan_id} deleted"}


def get_challan(challan_id):
    conn = get_connection()
    row  = conn.execute("SELECT * FROM challans WHERE id=?",
                        (challan_id,)).fetchone()
    conn.close()
    if row:
        c = dict(row)
        c["violations"]   = json.loads(c["violations"])
        c["fine_details"] = json.loads(c["fine_details"])
        return c
    return None


def get_all_challans():
    conn = get_connection()
    rows = conn.execute("""
        SELECT c.*, o.owner_name, o.phone, o.email
        FROM challans c
        LEFT JOIN vehicle_owners o ON c.plate_number = o.plate_number
        ORDER BY c.created_at DESC
    """).fetchall()
    conn.close()
    result = []
    for row in rows:
        c = dict(row)
        c["violations"]   = json.loads(c["violations"])
        c["fine_details"] = json.loads(c["fine_details"])
        result.append(c)
    return result


def get_dashboard_stats():
    conn = get_connection()
    s    = {}
    s["total_challans"]   = conn.execute("SELECT COUNT(*) FROM challans").fetchone()[0]
    s["pending_challans"] = conn.execute("SELECT COUNT(*) FROM challans WHERE status='pending'").fetchone()[0]
    s["sent_challans"]    = conn.execute("SELECT COUNT(*) FROM challans WHERE status='sent'").fetchone()[0]
    s["paid_challans"]    = conn.execute("SELECT COUNT(*) FROM challans WHERE status='paid'").fetchone()[0]
    s["total_owners"]     = conn.execute("SELECT COUNT(*) FROM vehicle_owners").fetchone()[0]
    s["total_fines"]      = conn.execute("SELECT COALESCE(SUM(total_fine),0) FROM challans").fetchone()[0]
    s["today_challans"]   = conn.execute(
        "SELECT COUNT(*) FROM challans WHERE DATE(created_at)=DATE('now','localtime')"
    ).fetchone()[0]
    conn.close()
    return s


if __name__ == "__main__":
    init_db()
