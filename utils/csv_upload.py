import csv
import io
from database import add_owner, get_connection


def process_csv(file_stream):
    """
    Process CSV file and bulk insert vehicle owners.
    Expected CSV columns:
    plate_number, owner_name, phone, email, address, vehicle_type
    Returns dict with success count, error count and error details.
    """
    results = {
        "success": 0,
        "failed": 0,
        "errors": [],
        "added": []
    }

    try:
        # Read CSV
        stream = io.StringIO(file_stream.read().decode("utf-8"))
        reader = csv.DictReader(stream)

        # Validate headers
        required_cols = {"plate_number", "owner_name"}
        headers = set(reader.fieldnames or [])
        missing = required_cols - headers
        if missing:
            return {
                "success": 0,
                "failed": 0,
                "errors": [f"Missing required columns: {', '.join(missing)}"],
                "added": []
            }

        for row_num, row in enumerate(reader, start=2):
            plate  = row.get("plate_number", "").strip().upper()
            name   = row.get("owner_name", "").strip()

            # Validate required fields
            if not plate or not name:
                results["failed"] += 1
                results["errors"].append(
                    f"Row {row_num}: plate_number and owner_name are required"
                )
                continue

            result = add_owner(
                plate_number = plate,
                owner_name   = name,
                phone        = row.get("phone", "").strip(),
                email        = row.get("email", "").strip(),
                address      = row.get("address", "").strip(),
                vehicle_type = row.get("vehicle_type", "unknown").strip()
            )

            if result["success"]:
                results["success"] += 1
                results["added"].append(plate)
            else:
                results["failed"] += 1
                results["errors"].append(f"Row {row_num} ({plate}): {result['message']}")

    except Exception as e:
        results["errors"].append(f"File processing error: {str(e)}")

    return results


def generate_sample_csv():
    """Generate a sample CSV string for download."""
    rows = [
        ["plate_number", "owner_name", "phone", "email", "address", "vehicle_type"],
        ["TS09AB1234", "Rajesh Kumar", "+91 98765 43210", "rajesh@email.com", "Hyderabad, TS", "motorcycle"],
        ["TS07CD5678", "Priya Sharma", "+91 87654 32109", "priya@email.com", "Secunderabad, TS", "car"],
        ["TS10EF9012", "Suresh Reddy", "+91 76543 21098", "suresh@email.com", "Warangal, TS", "car"],
        ["TS01GH3456", "Anita Singh", "+91 65432 10987", "anita@email.com", "Nizamabad, TS", "motorcycle"],
    ]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(rows)
    return output.getvalue()
