"""
Synchronize ALL local GateVision events to Prosper Smart Yard cloud for sarsite@gmail.com.
Key Requirements:
- Authentic login for sarsite@gmail.com (User ID: 01a0b4d0-d292-7f6c-8d5d-9e31503fcb4d)
- Uses unique VisitId prefix 'VISIT-SAR-' so visits belong strictly to sarsite@gmail.com
- Strict 1:1 event mapping:
    * ENTRY events are sent DIRECTLY and ONLY as GATE_IN
    * EXIT events are sent DIRECTLY and ONLY as GATE_OUT (zero prerequisite GATE_IN)
- Attaches only lane inspection proof photos (gate-in-1, gate-in-2, gate-out-1, gate-out-2)
- Excludes security guard / office photos (gate-in-3)
- Fully handles twin containers (FRONT and REAR)
"""

import sys
import os
import json
import base64
import time
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import requests

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)
from app.config.settings import Settings
from app.ocr.validator import extract_iso_size_code
from app.snapshots.snapshot_manager import SnapshotManager

settings = Settings()
snaps = SnapshotManager(settings.snapshot_directory)
session = requests.Session()

# 1. Login to Prosper Cloud with sarsite@gmail.com
print("=" * 70)
print("Logging in to Prosper Cloud as sarsite@gmail.com...")
auth_resp = session.post(
    "https://cfsapi.prosperassettracking.com/api/auth/login",
    json={"username": "sarsite@gmail.com", "password": "Test@123"},
    timeout=15
)
if auth_resp.status_code not in (200, 201):
    print(f"Authentication failed ({auth_resp.status_code}): {auth_resp.text}")
    sys.exit(1)

auth_data = auth_resp.json()
token = auth_data.get("accessToken") or auth_data.get("token")
parts = token.split(".")
claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
site_id = str(claims.get("site_ids", ""))
client_id = str(claims.get("client_ids", ""))
sarsite_uid = str(claims.get("sub") or claims.get("user_id") or "")
print(f"Auth Success! User: sarsite@gmail.com (UID: {sarsite_uid}) | Site ID: {site_id}")

get_headers = {
    "Authorization": f"Bearer {token}",
    "X-Site-Id": site_id,
    "X-Client-Id": client_id
}
post_headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
    "X-Site-Id": site_id,
    "X-Client-Id": client_id
}
api_url = f"{settings.prosper_api_base_url.rstrip('/')}/api/v1/gate/events"
visits_url = f"{settings.prosper_api_base_url.rstrip('/')}/api/v1/gate/visits"
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

# 2. Check current sarsite visits to avoid duplicates
print("\nFetching existing visits for sarsite@gmail.com...")
existing_sarsite_events = set()

page = 1
while True:
    try:
        res = session.get(f"{visits_url}?Page={page}&PageSize=100", headers=get_headers, timeout=15)
        if res.status_code != 200:
            break
        items = res.json().get("items", [])
        if not items:
            break
        for v in items:
            c = (v.get("containerNumber") or "").strip().upper()
            vid = v.get("visitId", "")
            for ev in v.get("events", []):
                et = (ev.get("eventType") or "").strip().upper()
                c_ev = (ev.get("detectedContainerNumber") or c).strip().upper()
                # Check if created by sarsite
                if ev.get("createdByUserId") == sarsite_uid:
                    existing_sarsite_events.add((c_ev, et))
                    if vid:
                        existing_sarsite_events.add(vid)
        if len(items) < 100:
            break
        page += 1
    except Exception as e:
        print(f"Warning fetching visits: {e}")
        break

print(f"Loaded {len(existing_sarsite_events)} existing event records for sarsite.")

# 3. Query all confirmed local events
db_path = os.path.join(backend_dir, "data", "gate.db")
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

local_events = cur.execute("""
    SELECT id, timestamp, container_number, event_type, status, cloud_synced,
           is_twin, container_number_2, iso_code_1, iso_code_2
    FROM gate_events
    WHERE container_number IS NOT NULL AND status = 'CONFIRMED'
    ORDER BY timestamp ASC
""").fetchall()

print(f"Total confirmed events in local DB: {len(local_events)}")

# Filter events: send all that aren't already recorded under sarsite
to_process = []
skipped_count = 0

for ev in local_events:
    cntr = (ev["container_number"] or "").strip().upper()
    direction = "GATE_OUT" if ev["event_type"] in ("EXIT", "OUT", "GATE_OUT") else "GATE_IN"
    
    if (cntr, direction) in existing_sarsite_events:
        skipped_count += 1
        continue
    
    to_process.append((ev, cntr, direction))

print(f"Skipping {skipped_count} events already verified under sarsite@gmail.com.")
print(f"Sending {len(to_process)} events to sarsite@gmail.com.")
print("=" * 70)

success_count = 0
failed_count = 0

for ev, cntr, ev_type in to_process:
    event_id = ev["id"]
    ts = ev["timestamp"]
    dt_obj = datetime.fromisoformat(ts.replace("Z", "+00:00")) if isinstance(ts, str) else datetime.now(timezone.utc)
    
    # Collect lane inspection proof snapshots
    snaps_rows = cur.execute("""
        SELECT camera_id, image_path, timestamp
        FROM snapshots
        WHERE event_id = ?
    """, (event_id,)).fetchall()

    proof_images = []
    for s_row in snaps_rows:
        cam_id = (s_row["camera_id"] or "").lower()
        # Exclude security guard cabin / office cameras (gate-in-3)
        if "gate-in-3" in cam_id or "office" in cam_id or "cabin" in cam_id:
            continue
        img_rel = s_row["image_path"]
        try:
            p = snaps.path(img_rel)
            if p and p.is_file():
                img_b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
                proof_images.append({
                    "ImageType": "FRONT",
                    "CameraId": s_row["camera_id"],
                    "CapturedAt": iso_utc(s_row["timestamp"]),
                    "Image": f"data:image/jpeg;base64,{img_b64}"
                })
        except Exception:
            pass

    # Prefer front / lane camera, take top 2
    proof_images.sort(key=lambda x: 0 if any(k in x["CameraId"].lower() for k in ("-1", "front")) else 1)
    proof_images = proof_images[:2]

    if not proof_images:
        print(f"[{event_id[:8]}] SKIP {cntr} ({ev_type}) - No lane inspection images available")
        failed_count += 1
        continue

    api_size, human_size = derive_size(ev["iso_code_1"] or "", "")
    is_twin = bool(ev["is_twin"]) and bool(ev["container_number_2"])
    container_payload = {
        "ContainerNumber": cntr,
        "ContainerNumberConfidence": 0.95,
        "Size": api_size,
        "ContainerSize": human_size,
        "SizeCode": ev["iso_code_1"] or None,
        "Position": "FRONT" if is_twin else "SINGLE",
        "Images": proof_images
    }
    containers_list = [container_payload]
    if is_twin:
        c2 = ev["container_number_2"].strip().upper()
        api_size_2, human_size_2 = derive_size(ev["iso_code_2"] or "", "")
        containers_list.append({
            "ContainerNumber": c2,
            "ContainerNumberConfidence": 0.95,
            "Size": api_size_2,
            "ContainerSize": human_size_2,
            "SizeCode": ev["iso_code_2"] or None,
            "Position": "REAR",
            "Images": proof_images
        })

    # Use unique sarsite visitId so it creates the visit directly under sarsite's account
    visit_id = f"VISIT-SAR-{dt_obj.strftime('%Y%m%d')}-{event_id[:8].upper()}"

    # Pure direct 1:1 POST: ENTRY -> GATE_IN only, EXIT -> GATE_OUT only!
    payload = {
        "VisitId": visit_id,
        "EventType": ev_type,
        "DeviceId": device_id,
        "CapturedAt": iso_utc(ts),
        "containerNumber": cntr,
        "Truck": {"TruckNumber": "NA"},
        "Driver": None,
        "Containers": containers_list
    }

    print(f"[{event_id[:8]}] Sending {ev_type:8s} for {cntr}...", end=" ", flush=True)
    try:
        res = session.post(api_url, json=payload, headers=post_headers, timeout=30)
        if res.status_code in (200, 201):
            res_data = res.json()
            cloud_ev_id = res_data.get("eventId") or res_data.get("visitDatabaseId")
            print(f"SUCCESS! (Cloud ID: {cloud_ev_id})")
            cur.execute("""
                UPDATE gate_events
                SET cloud_synced = 1, cloud_synced_at = ?, cloud_event_id = ?
                WHERE id = ?
            """, (iso_utc(datetime.now(timezone.utc)), str(cloud_ev_id), event_id))
            conn.commit()
            existing_sarsite_events.add((cntr, ev_type))
            success_count += 1
        elif res.status_code == 409:
            print(f"Already exists on cloud (409 Conflict) -> Marked synced.")
            cur.execute("UPDATE gate_events SET cloud_synced = 1 WHERE id = ?", (event_id,))
            conn.commit()
            success_count += 1
        else:
            print(f"FAILED ({res.status_code}): {res.text[:120]}")
            failed_count += 1
    except Exception as ex:
        print(f"ERROR: {ex}")
        failed_count += 1

    time.sleep(1)

print("\n" + "=" * 70)
print(f"Sync Complete! Success: {success_count} | Skipped: {skipped_count} | Failed: {failed_count}")
print("=" * 70)
