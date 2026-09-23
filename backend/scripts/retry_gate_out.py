import sys, json, base64, requests, time
sys.path.insert(0, ".")
from app.config.settings import Settings
from app.database.database import Database
from app.database.repositories.gate import GateRepository
from app.ocr.validator import extract_iso_size_code
from app.snapshots.snapshot_manager import SnapshotManager
from datetime import datetime, timezone

s = Settings()
db = Database(s.database_url.get_secret_value())
repo = GateRepository(db)
snaps = SnapshotManager(s.snapshot_directory)

r = requests.post("https://cfsapi.prosperassettracking.com/api/auth/login",
    json={"username": "sarsite@gmail.com", "password": "Test@123"}, timeout=10)
token = r.json().get("accessToken") or r.json().get("token")
parts = token.split(".")
claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
site_id = str(claims.get("site_ids", ""))
client_id = str(claims.get("client_ids", ""))
headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json", "X-Site-Id": site_id, "X-Client-Id": client_id}
endpoint = s.prosper_api_base_url.rstrip("/") + "/api/v1/gate/events"

failed_ids = ["0b2af679", "a54b201c", "1f22601e", "175222cf"]
all_evs = repo.events(limit=200)["items"]
exits = [e for e in all_evs if (e.get("event_type") or "").upper() in ("EXIT","OUT","GATE_OUT") and e.get("container_number")]

for summary in exits:
    eid8 = summary["id"][:8]
    if eid8 not in failed_ids:
        continue
    event = repo.event(summary["id"])
    c1 = event.get("container_number")
    ts = event.get("timestamp", "")
    visit_id = "VISIT-20260922-" + summary["id"][:8].upper()

    imgs = []
    for s_rec in event.get("snapshots", []):
        ip = s_rec.get("image_path") or ""
        cam = (s_rec.get("camera_id") or "").lower()
        if not (ip.endswith("_rear.jpg") or any(k in cam for k in ("-3","rear"))):
            p = snaps.path(ip)
            if p and p.is_file():
                imgs.append({"ImageType": "FRONT", "CameraId": s_rec.get("camera_id", "gate-out-1"),
                    "CapturedAt": ts, "Image": "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()})

    _, human_sz = extract_iso_size_code(event.get("iso_code_1") or "")
    api_sz = "20" if human_sz and "20" in human_sz else "40"

    payload = {"VisitId": visit_id, "EventType": "GATE_OUT", "DeviceId": "EDGE-GATE-01",
        "CapturedAt": ts, "Truck": {"TruckNumber": "NA"}, "Driver": None,
        "Containers": [{"ContainerNumber": c1, "ContainerNumberConfidence": 0.95, "Size": api_sz,
            "Position": "SINGLE", "Images": imgs[:3]}]}

    print("Retrying GATE_OUT for " + c1 + " (" + eid8 + ") ...")
    time.sleep(3)
    r2 = requests.post(endpoint, json=payload, headers=headers, timeout=25)
    print("  Status: " + str(r2.status_code))
    try:
        body = r2.json()
        print("  Body: " + json.dumps(body)[:250])
    except Exception:
        print("  Body: " + r2.text[:250])
