"""
utils/challan_generator.py — FIXED
=====================================
BUG FIX: ₹ rendered as ■ in PDF.
ReportLab's built-in Helvetica has no ₹ glyph.
Fix: auto-detect and register a Unicode TTF font (DejaVu/Arial/Calibri).
Fallback: "Rs." if no suitable font found on the machine.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, Image, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from config import CHALLANS_DIR

# ── Unicode font detection ────────────────────────────────────────────────────
RUPEE_FONT = "Helvetica"
RUPEE_BOLD = "Helvetica-Bold"
RUPEE_SYM  = "Rs."

_CANDIDATES = [
    r"C:\Windows\Fonts\DejaVuSans.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]
for _fp in _CANDIDATES:
    if os.path.exists(_fp):
        try:
            pdfmetrics.registerFont(TTFont("UniFont", _fp))
            RUPEE_FONT = "UniFont"
            RUPEE_BOLD = "UniFont"   # TTF bold via same file (weight handled by PDF viewer)
            RUPEE_SYM  = "₹"
            print(f"[ChallanPDF] Unicode font: {_fp}  ₹ enabled")
            break
        except Exception as e:
            continue

if RUPEE_SYM == "Rs.":
    print("[ChallanPDF] No Unicode font found — using 'Rs.' instead of ₹")


def _rs(amount):
    return f"{RUPEE_SYM}{amount:,}"


def generate_challan_pdf(challan_id, plate_number, violations, fine_details,
                          total_fine, owner=None, image_path=None,
                          annotated_image=None) -> str:

    os.makedirs(CHALLANS_DIR, exist_ok=True)
    pdf_path = os.path.join(CHALLANS_DIR, f"challan_{challan_id}.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    styles  = getSampleStyleSheet()
    content = []
    RF, RB  = RUPEE_FONT, RUPEE_BOLD

    def S(name, **kw):
        kw.setdefault("fontName", RF)
        return ParagraphStyle(name, parent=styles["Normal"], **kw)

    # Header
    content += [
        Paragraph("TRAFFIC VIOLATION CHALLAN",
                  S("H", fontSize=20, fontName=RB,
                    textColor=colors.HexColor("#1a237e"),
                    spaceAfter=4, alignment=TA_CENTER)),
        Paragraph("Government of India — Traffic Police Department",
                  S("Sub", fontSize=10, textColor=colors.HexColor("#424242"),
                    alignment=TA_CENTER)),
        Spacer(1, 6),
        HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a237e")),
        Spacer(1, 10),
    ]

    # Challan info table
    info = Table([
        ["Challan No.", f"#{challan_id:06d}",
         "Date & Time", datetime.now().strftime("%d/%m/%Y %H:%M")],
        ["Vehicle Plate", plate_number or "UNKNOWN", "Status", "PENDING PAYMENT"],
    ], colWidths=[100,160,100,160])
    info.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#e8eaf6")),
        ("BACKGROUND",(2,0),(2,-1),colors.HexColor("#e8eaf6")),
        ("FONTNAME",  (0,0),(-1,-1),RF),
        ("FONTNAME",  (0,0),(0,-1), RB),
        ("FONTNAME",  (2,0),(2,-1), RB),
        ("FONTSIZE",  (0,0),(-1,-1),10),
        ("GRID",      (0,0),(-1,-1),0.5,colors.HexColor("#9fa8da")),
        ("PADDING",   (0,0),(-1,-1),8),
    ]))
    content += [info, Spacer(1,12)]

    # Owner
    content.append(Paragraph("VEHICLE OWNER DETAILS",
                              S("SH",fontSize=11,fontName=RB,
                                textColor=colors.HexColor("#1a237e"),spaceAfter=6)))
    odata = [
        ["Owner Name", owner.get("owner_name","N/A") if owner else "Not registered in system"],
        ["Phone",      owner.get("phone","N/A")      if owner else "N/A"],
        ["Email",      owner.get("email","N/A")      if owner else "N/A"],
        ["Address",    owner.get("address","N/A")    if owner else "N/A"],
    ]
    ot = Table(odata, colWidths=[120,400])
    ot.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#f3f4f6")),
        ("FONTNAME",  (0,0),(0,-1),RB), ("FONTNAME",(1,0),(1,-1),RF),
        ("FONTSIZE",  (0,0),(-1,-1),10),
        ("GRID",      (0,0),(-1,-1),0.5,colors.HexColor("#d1d5db")),
        ("PADDING",   (0,0),(-1,-1),7),
    ]))
    content += [ot, Spacer(1,12)]

    # Violations
    content.append(Paragraph("VIOLATIONS DETECTED",
                              S("VH",fontSize=11,fontName=RB,
                                textColor=colors.HexColor("#b71c1c"),spaceAfter=6)))
    VLABELS = {
        "no_helmet":     "Riding without helmet",
        "triple_riding": "Triple riding (more than 2 riders)",
        "no_seatbelt":   "Driving without seatbelt",
        "wrong_lane":    "Riding in wrong lane",
    }
    fdata = [["#","Violation","Section","Fine Amount"]]
    for i,(v,amt) in enumerate(fine_details.items(),1):
        fdata.append([str(i), VLABELS.get(v,v), "MV Act", _rs(amt)])
    fdata.append(["","","TOTAL FINE", _rs(total_fine)])

    ft = Table(fdata, colWidths=[30,270,100,120])
    ft.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,0), colors.HexColor("#b71c1c")),
        ("TEXTCOLOR",    (0,0),(-1,0), colors.white),
        ("FONTNAME",     (0,0),(-1,0), RB),
        ("FONTNAME",     (0,1),(-1,-1),RF),
        ("FONTNAME",     (2,-1),(-1,-1),RB),
        ("FONTSIZE",     (0,0),(-1,-1),10),
        ("BACKGROUND",   (0,-1),(-1,-1),colors.HexColor("#ffebee")),
        ("TEXTCOLOR",    (2,-1),(-1,-1),colors.HexColor("#b71c1c")),
        ("GRID",         (0,0),(-1,-1),0.5,colors.HexColor("#ef9a9a")),
        ("PADDING",      (0,0),(-1,-1),8),
        ("ROWBACKGROUNDS",(0,1),(-1,-2),
         [colors.white,colors.HexColor("#fff8f8")]),
        ("ALIGN",        (0,0),(0,-1),"CENTER"),
        ("ALIGN",        (3,0),(3,-1),"RIGHT"),
    ]))
    content += [ft, Spacer(1,12)]

    # Image
    img_to_show = annotated_image or image_path
    if img_to_show and os.path.exists(img_to_show):
        content.append(Paragraph("VIOLATION IMAGE",
                                  S("IH",fontSize=11,fontName=RB,
                                    textColor=colors.HexColor("#1a237e"),spaceAfter=6)))
        try:
            im = Image(img_to_show, width=5*inch, height=3*inch)
            im.hAlign = "CENTER"
            content += [im, Spacer(1,12)]
        except Exception as e:
            print(f"[ChallanPDF] Image error: {e}")

    # Payment
    content += [
        HRFlowable(width="100%",thickness=1,color=colors.HexColor("#9fa8da")),
        Spacer(1,8),
        Paragraph("PAYMENT INSTRUCTIONS",
                  S("PH",fontSize=11,fontName=RB,
                    textColor=colors.HexColor("#1a237e"),spaceAfter=6)),
        Paragraph(
            "1. Pay online at <b>echallan.parivahan.gov.in</b><br/>"
            "2. Pay at nearest Traffic Police Station<br/>"
            "3. Non-payment within 30 days may result in further legal action.<br/>"
            f"4. Total Amount Due: <b>{_rs(total_fine)}</b>",
            S("P2",fontSize=10,leading=18)),
        Spacer(1,20),
        HRFlowable(width="100%",thickness=1,color=colors.HexColor("#9fa8da")),
        Spacer(1,6),
        Paragraph("This is a computer-generated challan. | Traffic Challan System",
                  S("Ft",fontSize=8,textColor=colors.gray,alignment=TA_CENTER)),
    ]

    doc.build(content)
    print(f"Challan PDF generated: {pdf_path}")
    return pdf_path