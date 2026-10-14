import sys
import os
import shutil
import sqlite3
from pathlib import Path

def delete_events(event_ids: list[str], db_path: str = "data/gate.db"):
    if not event_ids:
        print("No event IDs provided.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for eid in event_ids:
        det_ids = [r[0] for r in cur.execute("SELECT id FROM detections WHERE event_id = ?", (eid,)).fetchall()]
        if det_ids:
            placeholders = ','.join('?' for _ in det_ids)
            cur.execute(f"DELETE FROM ocr_results WHERE detection_id IN ({placeholders})", det_ids)

        cur.execute("DELETE FROM detections WHERE event_id = ?", (eid,))
        cur.execute("DELETE FROM snapshots WHERE event_id = ?", (eid,))
        cur.execute("DELETE FROM system_logs WHERE event_id = ?", (eid,))
        cur.execute("DELETE FROM gate_events WHERE id = ?", (eid,))

        snap_dir = Path("snapshots") / eid
        if snap_dir.exists():
            shutil.rmtree(snap_dir, ignore_errors=True)

    conn.commit()
    conn.close()

    for spool_path in ["data/ocr_spool/jobs.sqlite3", "uploads/ocr-spool/jobs.sqlite3"]:
        if os.path.exists(spool_path):
            sconn = sqlite3.connect(spool_path)
            scur = sconn.cursor()
            for eid in event_ids:
                scur.execute("DELETE FROM jobs WHERE event_id = ?", (eid,))
            sconn.commit()
            sconn.close()

    print(f"Successfully deleted {len(event_ids)} event(s): {event_ids[:5]}...")

def clean_unassigned_events(lane: str = "unassigned-gate-out-3", db_path: str = "data/gate.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    ids = [r[0] for r in cur.execute("SELECT id FROM gate_events WHERE gate_id = ?", (lane,)).fetchall()]
    conn.close()

    print(f"Found {len(ids)} events with gate_id '{lane}'")
    if ids:
        delete_events(ids, db_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all-unassigned":
            clean_unassigned_events()
        else:
            delete_events(sys.argv[1:])
    else:
        # Default cleanup list
        default_ids = ["957e983e-1ba4-4e55-85b9-744f4ee69066"]
        delete_events(default_ids)
