"""
Sync today's (2026-09-23) confirmed events to Prosper Smart Yard cloud
using abhay@prosperinfotech.com credentials.
Ensures only primary/lane container inspection photos are sent (no security/guard cabin photos).
"""
import sys
import json
import base64
import requests
import time
import uuid
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
from app.config.settings import Settings
from app.ocr.validator import extract_iso_size_code
from app.snapshots.snapshot_manager import SnapshotManager

settings = Settings()
snaps = SnapshotManager(settings.snapshot_directory)
session = requests.Session()

# 1. Login with abhay@prosperinfotech.com
print("Logging in to Prosper Cloud...")
auth_resp = session.post(
    "https://cfsapi.prosperassettracking.com/api/auth/login",
    json={"username": "abhay@prosperinfotech.com", "password": "Abhi@123"},
    timeout=10
)
if auth_resp.status_code not in (200, 201):
    print("Authentication failed:", auth_resp.status_code, auth_resp.text)
    sys.exit(1)

token = auth_resp.json().get("accessToken") or auth_resp.json().get("token")
parts = token.split(".")
claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
site_id = str(claims.get("site_ids", ""))
client_id = str(claims.get("client_ids", ""))
print(f"Authenticated as abhay@prosperinfotech.com | Site ID: {site_id}")

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
    "X-Site-Id": site_id,
    "X-Client-Id": client_id
}
api_url = f"{settings.prosper_api_base_url.rstrip('/')}/api/v1/gate/events"
device_id = getattr(settings, "prosper_device_uuid", None) or "EDGE-GATE-01"

def iso_utc(val):
    if isinstance(val, str):
        try:
            d = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + str(d.microsecond // 1000).zfill(3) + "Z"
        except Exception:
            return val
    if isinstance(val, datetime):
        d = val if val.tzinfo else val.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + str(d.microsecond // 1000).zfill(3) + "Z"
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

def derive_size(iso_code, container_size):
    cs = container_size or ""
    if not cs and iso_code:
        _, cs = extract_iso_size_code(iso_code)
        cs = cs or ""
    if "20" in cs or (iso_code or "").startswith("2"):
        return "20", cs or "20 FT"
    if "45" in cs or (iso_code or "").startswith("L"):
        return "45", cs or "45 FT HC"
    return "40", cs or "40 FT"

# 2. Query today's confirmed events
con = sqlite3.connect("data/gate.db")
con.row_factory = sqlite3.Row
cur = con.cursor()

today_events = cur.execute("""
    SELECT id, timestamp, container_number, event_type, status, cloud_synced,
           is_twin, container_number_2, iso_code_1, iso_code_2
    FROM gate_events
    WHERE timestamp LIKE '2026-09-23%' AND status = 'CONFIRMED' AND container_number IS NOT NULL
    ORDER BY timestamp ASC
""").fetchall()

print(f"\nFound {len(today_events)} confirmed events for today (2026-09-23):")
print("-" * 65)

success_count = 0
failed_count = 0

for ev in today_events:
    event_id = ev["id"]
    cntr = ev["container_number"]
    ev_type = "GATE_OUT" if ev["event_type"] in ("EXIT", "OUT", "GATE_OUT") else "GATE_IN"
    ts = ev["timestamp"]
    visit_id = f"VISIT-{event_id[:8].upper()}-{int(time.time()) % 100000}"

    # Fetch snapshots for this event
    snaps_rows = cur.execute("""
        SELECT camera_id, image_path, timestamp
        FROM snapshots
        WHERE event_id = ?
    """, (event_id,)).fetchall()

    # Filter snapshots: only take container lane inspection images (gate-in-1, gate-out-1, gate-out-2)
    # Exclude gate-in-3 (security guard cabin/office)
    proof_images = []
    for s_row in snaps_rows:
        cam_id = (s_row["camera_id"] or "").lower()
        if "gate-in-3" in cam_id or "office" in cam_id or "cabin" in cam_id:
            continue  # SKIP office / guard cabin camera!

        img_rel = s_row["image_path"]
        try:
            p = snaps.path(img_rel)
            if p.is_file():
                img_b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
                proof_images.append({
                    "ImageType": "FRONT",
                    "CameraId": s_row["camera_id"],
                    "CapturedAt": iso_utc(s_row["timestamp"]),
                    "Image": f"data:image/jpeg;base64,{img_b64}"
                })
        except Exception as e:
            pass

    # Prefer gate-in-1 or gate-out-1 / gate-out-2
    proof_images.sort(key=lambda x: 0 if any(k in x["CameraId"].lower() for k in ("-1", "front")) else 1)
    # Keep at most 2 best images
    proof_images = proof_images[:2]

    if not proof_images:
        print(f"[{event_id[:8]}] SKIP {cntr} ({ev_type}) - No valid lane inspection image")
        continue

    api_size, human_size = derive_size(ev["iso_code_1"] or "", "")

    container_payload = {
        "ContainerNumber": cntr,
        "ContainerNumberConfidence": 0.95,
        "Size": api_size,
        "ContainerSize": human_size,
        "SizeCode": ev["iso_code_1"] or None,
        "Position": "SINGLE",
        "Images": proof_images
    }

    event_payload = {
        "VisitId": visit_id,
        "EventType": ev_type,
        "DeviceId": device_id,
        "CapturedAt": iso_utc(ts),
        "containerNumber": cntr,
        "Truck": {"TruckNumber": "NA"},
        "Driver": None,
        "Containers": [container_payload]
    }

    print(f"[{event_id[:8]}] Posting {ev_type} for {cntr} (cams: {[img['CameraId'] for img in proof_images]})...", end=" ")
    
    try:
        res = session.post(api_url, json=event_payload, headers=headers, timeout=25)
        if res.status_code in (200, 201):
            res_data = res.json()
            cloud_ev_id = res_data.get("eventId") or res_data.get("visitDatabaseId")
            print(f"OK! (EventId: {cloud_ev_id})")
            cur.execute("""
                UPDATE gate_events
                SET cloud_synced = 1, cloud_synced_at = ?, cloud_event_id = ?
                WHERE id = ?
            """, (iso_utc(datetime.now(timezone.utc)), str(cloud_ev_id), event_id))
            con.commit()
            success_count += 1
        elif res.status_code == 409:
            print("Already exists (409 Conflict) - Marked synced.")
            cur.execute("""
                UPDATE gate_events
                SET cloud_synced = 1, cloud_synced_at = ?
                WHERE id = ?
            """, (iso_utc(datetime.now(timezone.utc)), event_id))
            con.commit()
            success_count += 1
        else:
            print(f"FAILED ({res.status_code}): {res.text[:120]}")
            failed_count += 1
    except Exception as exc:
        print(f"ERROR: {str(exc)[:100]}")
        failed_count += 1

    time.sleep(1)

print("=" * 65)
print(f"Summary: {success_count} successfully posted, {failed_count} failed.")
