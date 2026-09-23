import sys, json, base64, requests, time, uuid
sys.path.insert(0, ".")
from app.config.settings import Settings
from app.database.database import Database
from app.database.repositories.gate import GateRepository
from app.ocr.validator import extract_iso_size_code
from app.snapshots.snapshot_manager import SnapshotManager
from datetime import datetime, timezone, timedelta

s = Settings()
db = Database(s.database_url.get_secret_value())
repo = GateRepository(db)
snaps = SnapshotManager(s.snapshot_directory)
session = requests.Session()

# Auth with new credentials
r = session.post("https://cfsapi.prosperassettracking.com/api/auth/login",
    json={"username": "abhay@prosperinfotech.com", "password": "Abhi@123"}, timeout=10)
data = r.json()
token = data.get("accessToken") or data.get("token")
parts = token.split(".")
claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
site_id = str(claims.get("site_ids", ""))
client_id = str(claims.get("client_ids", ""))
headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json",
           "X-Site-Id": site_id, "X-Client-Id": client_id}
endpoint = s.prosper_api_base_url.rstrip("/") + "/api/v1/gate/events"
device_id = getattr(s, "prosper_device_uuid", None) or "EDGE-GATE-01"
print("Auth OK  user=abhay@prosperinfotech.com  site=" + site_id)
print()

def iso_utc(dt):
    if isinstance(dt, str):
        try:
            d = datetime.fromisoformat(dt.replace("Z", "+00:00"))
            return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + str(d.microsecond//1000).zfill(3) + "Z"
        except Exception:
            return dt
    if isinstance(dt, datetime):
        d = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + str(d.microsecond//1000).zfill(3) + "Z"
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

all_evs = repo.events(limit=200)["items"]
exits = [e for e in all_evs if (e.get("event_type") or "").upper() in ("EXIT","OUT","GATE_OUT") and e.get("container_number")]
print("Found " + str(len(exits)) + " EXIT events.")
print("=" * 60)

synced = 0
failed = 0

for summary in exits:
    event = repo.event(summary["id"])
    c1 = event.get("container_number")
    ts = event.get("timestamp", "")
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.now(timezone.utc)

    # Fresh unique VisitId to avoid 409 conflict from previous run
    fresh_visit_id = "VISIT-" + dt.strftime("%Y%m%d") + "-" + str(uuid.uuid4())[:8].upper()
    arrival_dt = dt - timedelta(hours=8)

    event_snaps = event.get("snapshots", [])
    front_imgs = []
    for s_rec in event_snaps:
        ip = s_rec.get("image_path") or ""
        cam = (s_rec.get("camera_id") or "").lower()
        if not (ip.endswith("_rear.jpg") or any(k in cam for k in ("-3","rear"))):
            p = snaps.path(ip)
            if p and p.is_file():
                front_imgs.append({"ImageType": "FRONT", "CameraId": s_rec.get("camera_id", "gate-out-1"),
                    "CapturedAt": iso_utc(s_rec.get("timestamp") or ts),
                    "Image": "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()})

    if not front_imgs:
        print("[" + summary["id"][:8] + "] SKIP " + c1 + " - no images")
        failed += 1
        continue

    api_sz, human_sz = derive_size(event.get("iso_code_1") or "", event.get("container_size") or "")
    container_entry = {"ContainerNumber": c1, "ContainerNumberConfidence": 0.95,
        "Size": api_sz, "ContainerSize": human_sz or None,
        "SizeCode": event.get("iso_code_1") or None,
        "Position": "SINGLE", "Images": front_imgs[:3]}

    print("[" + summary["id"][:8] + "]  " + c1 + "  size=" + api_sz + "(" + human_sz + ")  VisitId=" + fresh_visit_id)

    # GATE_IN
    in_payload = {"VisitId": fresh_visit_id, "EventType": "GATE_IN", "DeviceId": device_id,
        "CapturedAt": iso_utc(arrival_dt), "Truck": {"TruckNumber": "NA"}, "Driver": None,
        "Containers": [container_entry]}
    r1 = session.post(endpoint, json=in_payload, headers=headers, timeout=25)
    if r1.status_code in (200, 201):
        d1 = r1.json()
        print("  GATE_IN  OK 200  eventId=" + d1.get("eventId",""))
    else:
        body1 = ""
        try: body1 = json.dumps(r1.json())[:150]
        except: body1 = r1.text[:150]
        print("  GATE_IN  FAIL " + str(r1.status_code) + "  " + body1)
        if r1.status_code not in (409,):
            failed += 1
            continue

    time.sleep(2)

    # GATE_OUT
    out_payload = {"VisitId": fresh_visit_id, "EventType": "GATE_OUT", "DeviceId": device_id,
        "CapturedAt": iso_utc(dt), "Truck": {"TruckNumber": "NA"}, "Driver": None,
        "Containers": [container_entry]}
    r2 = session.post(endpoint, json=out_payload, headers=headers, timeout=25)
    if r2.status_code in (200, 201):
        d2 = r2.json()
        print("  GATE_OUT OK 200  eventId=" + d2.get("eventId",""))
        synced += 1
    else:
        body2 = ""
        try: body2 = json.dumps(r2.json())[:150]
        except: body2 = r2.text[:150]
        print("  GATE_OUT FAIL " + str(r2.status_code) + "  " + body2)
        failed += 1

    time.sleep(1)

print()
print("=" * 60)
print("DONE.  Synced(GATE_OUT OK): " + str(synced) + "  Failed: " + str(failed))
print("=" * 60)
