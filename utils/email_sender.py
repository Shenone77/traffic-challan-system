import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from config import EMAIL_HOST, EMAIL_PORT, EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_SUBJECT


def send_challan_email(to_email, owner_name, challan_id, plate_number,
                        violations, fine_details, total_fine, pdf_path=None) -> dict:
    """Send challan details via email with PDF attachment."""

    violation_labels = {
        "no_helmet"     : "Riding without helmet",
        "triple_riding" : "Triple riding (more than 2 riders)",
        "no_seatbelt"   : "Driving without seatbelt",
        "wrong_lane"    : "Riding in wrong lane",
    }

    # Build violations HTML
    violations_html = ""
    for violation, amount in fine_details.items():
        violations_html += f"""
        <tr>
            <td style="padding:8px;border:1px solid #ddd;">
                {violation_labels.get(violation, violation)}
            </td>
            <td style="padding:8px;border:1px solid #ddd;text-align:right;">
                ₹{amount:,}
            </td>
        </tr>"""

    html_body = f"""
    <html>
    <body style="font-family:Arial,sans-serif;max-width:600px;margin:auto;">
        <div style="background:#1a237e;padding:20px;text-align:center;">
            <h1 style="color:white;margin:0;">🚦 Traffic Violation Challan</h1>
            <p style="color:#c5cae9;margin:5px 0;">Government Traffic Police Department</p>
        </div>

        <div style="padding:20px;background:#f5f5f5;">
            <p>Dear <b>{owner_name}</b>,</p>
            <p>A traffic violation has been recorded for your vehicle.
               Please find the challan details below:</p>

            <table style="width:100%;border-collapse:collapse;margin:15px 0;">
                <tr style="background:#e8eaf6;">
                    <td style="padding:8px;border:1px solid #ddd;font-weight:bold;">
                        Challan No.
                    </td>
                    <td style="padding:8px;border:1px solid #ddd;">
                        #{challan_id:06d}
                    </td>
                </tr>
                <tr>
                    <td style="padding:8px;border:1px solid #ddd;font-weight:bold;">
                        Vehicle Plate
                    </td>
                    <td style="padding:8px;border:1px solid #ddd;">
                        <b>{plate_number}</b>
                    </td>
                </tr>
            </table>

            <h3 style="color:#b71c1c;">Violations Detected:</h3>
            <table style="width:100%;border-collapse:collapse;">
                <tr style="background:#b71c1c;color:white;">
                    <th style="padding:8px;border:1px solid #ddd;text-align:left;">
                        Violation
                    </th>
                    <th style="padding:8px;border:1px solid #ddd;text-align:right;">
                        Fine Amount
                    </th>
                </tr>
                {violations_html}
                <tr style="background:#ffebee;">
                    <td style="padding:8px;border:1px solid #ddd;font-weight:bold;">
                        TOTAL FINE
                    </td>
                    <td style="padding:8px;border:1px solid #ddd;
                               font-weight:bold;text-align:right;color:#b71c1c;">
                        ₹{total_fine:,}
                    </td>
                </tr>
            </table>

            <div style="background:#fff3e0;padding:15px;margin:15px 0;
                        border-left:4px solid #ff9800;">
                <h4 style="margin:0 0 8px;color:#e65100;">Payment Instructions:</h4>
                <ul style="margin:0;padding-left:20px;">
                    <li>Pay online at <a href="https://echallan.parivahan.gov.in">
                        echallan.parivahan.gov.in</a></li>
                    <li>Pay at nearest Traffic Police Station</li>
                    <li>Non-payment within 30 days may result in legal action</li>
                </ul>
            </div>

            <p style="color:#666;font-size:12px;">
                This is an auto-generated email. Please find the challan PDF attached.
            </p>
        </div>

        <div style="background:#1a237e;padding:10px;text-align:center;">
            <p style="color:#c5cae9;margin:0;font-size:12px;">
                Traffic Challan System | Auto-generated
            </p>
        </div>
    </body>
    </html>
    """

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"{EMAIL_SUBJECT} — Challan #{challan_id:06d} | {plate_number}"
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = to_email

        msg.attach(MIMEText(html_body, "html"))

        # Attach PDF if exists
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_attachment = MIMEApplication(f.read(), _subtype="pdf")
                pdf_attachment.add_header(
                    "Content-Disposition", "attachment",
                    filename=f"challan_{challan_id}.pdf"
                )
                msg.attach(pdf_attachment)

        # Send email
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, to_email, msg.as_string())

        return {"success": True, "message": f"Email sent to {to_email}"}

    except Exception as e:
        return {"success": False, "message": str(e)}
