import sys, json, base64, requests, time, uuid
sys.path.insert(0, ".")
from app.config.settings import Settings
from app.database.database import Database
from app.database.repositories.gate import GateRepository
from app.snapshots.snapshot_manager import SnapshotManager
from datetime import datetime, timezone, timedelta

s = Settings()
db = Database(s.database_url.get_secret_value())
repo = GateRepository(db)
snaps = SnapshotManager(s.snapshot_directory)
sess = requests.Session()

r = sess.post("https://cfsapi.prosperassettracking.com/api/auth/login",
    json={"username": "abhay@prosperinfotech.com", "password": "Abhi@123"}, timeout=10)
token = r.json().get("accessToken") or r.json().get("token")
parts = token.split(".")
claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
site_id = str(claims.get("site_ids", ""))
client_id = str(claims.get("client_ids", ""))
headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json",
           "X-Site-Id": site_id, "X-Client-Id": client_id}
endpoint = s.prosper_api_base_url.rstrip("/") + "/api/v1/gate/events"

all_evs = repo.events(limit=200)["items"]
exits = [e for e in all_evs if (e.get("event_type") or "").upper() in ("EXIT","OUT","GATE_OUT") and e.get("container_number")]
ev = repo.event(exits[0]["id"])
c1 = ev.get("container_number")
ts = ev.get("timestamp", "")
dt_obj = datetime.fromisoformat(ts.replace("Z", "+00:00"))
arrival = (dt_obj - timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
departure = dt_obj.strftime("%Y-%m-%dT%H:%M:%S.000Z")

imgs = []
for s_rec in ev.get("snapshots", []):
    ip = s_rec.get("image_path") or ""
    cam = (s_rec.get("camera_id") or "").lower()
    if not (ip.endswith("_rear.jpg") or any(k in cam for k in ("-3","rear"))):
        p = snaps.path(ip)
        if p and p.is_file():
            imgs.append({"ImageType": "FRONT", "CameraId": s_rec.get("camera_id", "gate-out-1"),
                "CapturedAt": ts, "Image": "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()})

visit_id = "VISIT-SCHEMA-TEST-" + str(uuid.uuid4())[:6].upper()
container = {"ContainerNumber": c1, "ContainerNumberConfidence": 0.95, "Size": "20", "Position": "SINGLE", "Images": imgs[:2]}

print("Testing GATE_OUT schema for container: " + c1)
print("VisitId: " + visit_id)
print()

# GATE_IN
r1 = sess.post(endpoint, json={"VisitId": visit_id, "EventType": "GATE_IN", "DeviceId": "EDGE-GATE-01",
    "CapturedAt": arrival, "Truck": {"TruckNumber": "NA"}, "Driver": None, "Containers": [container]},
    headers=headers, timeout=25)
print("GATE_IN status: " + str(r1.status_code))
r1_data = r1.json()
visit_db_id = r1_data.get("visitDatabaseId", "")
print("visitDatabaseId: " + visit_db_id)
print()
time.sleep(3)

# Try several GATE_OUT payload variations
attempts = [
    ("With visitDatabaseId", {"VisitId": visit_id, "visitDatabaseId": visit_db_id}),
    ("With VisitDatabaseId caps", {"VisitId": visit_id, "VisitDatabaseId": visit_db_id}),
    ("With id field", {"VisitId": visit_id, "id": visit_db_id}),
    ("VisitId = visitDatabaseId", {"VisitId": visit_db_id}),
    ("No VisitId just dbId", {"id": visit_db_id}),
    ("Original no extras", {"VisitId": visit_id}),
]

for label, extra_fields in attempts:
    payload = {"EventType": "GATE_OUT", "DeviceId": "EDGE-GATE-01",
        "CapturedAt": departure, "Truck": {"TruckNumber": "NA"}, "Driver": None, "Containers": [container]}
    payload.update(extra_fields)
    print("--- " + label + " ---")
    r2 = sess.post(endpoint, json=payload, headers=headers, timeout=25)
    print("  Status: " + str(r2.status_code))
    try:
        body = r2.json()
        print("  Body: " + json.dumps({k:v for k,v in body.items() if k != "images"})[:250])
    except Exception:
        print("  Body: " + r2.text[:250])
    if r2.status_code in (200, 201):
        print("  *** SUCCESS! ***")
        break
    time.sleep(2)
