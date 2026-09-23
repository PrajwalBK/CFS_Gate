import sqlite3

def clean():
    conn = sqlite3.connect("data/gate.db")
    cur = conn.cursor()
    ids = ("c9207caa-17c3-460c-b04d-52a95c3c3a3a", "a94219e2-7984-411f-b772-1988db94d83a")
    for eid in ids:
        cur.execute("DELETE FROM detections WHERE event_id = ?", (eid,))
        cur.execute("DELETE FROM snapshots WHERE event_id = ?", (eid,))
        cur.execute("DELETE FROM gate_events WHERE id = ?", (eid,))
    conn.commit()
    print(f"Successfully cleaned invalid events: {ids}")

if __name__ == "__main__":
    clean()
