import os
import shutil
import sqlite3
from pathlib import Path

def clear_all():
    base_dir = Path(r"d:\ai-video-based-inventory\ai-video-based-inventory\backend")
    db_path = base_dir / "data" / "gate.db"
    spool_db_path = base_dir / "uploads" / "ocr-spool" / "jobs.sqlite3"
    snapshots_dir = base_dir / "snapshots"
    spool_images_dir = base_dir / "uploads" / "ocr-spool" / "images"

    # 1. Clear gate.db tables
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM system_logs")
        cur.execute("DELETE FROM ocr_results")
        cur.execute("DELETE FROM snapshots")
        cur.execute("DELETE FROM detections")
        cur.execute("DELETE FROM gate_events")
        conn.commit()
        cur.execute("VACUUM")
        conn.commit()
        conn.close()
        print("Cleared gate.db tables (gate_events, detections, snapshots, ocr_results, system_logs)")

    # 2. Clear OCR spool database
    if spool_db_path.exists():
        conn = sqlite3.connect(spool_db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM jobs")
        conn.commit()
        cur.execute("VACUUM")
        conn.commit()
        conn.close()
        print("Cleared ocr-spool jobs.sqlite3")

    # 3. Clear snapshots directory
    if snapshots_dir.exists():
        for item in os.listdir(snapshots_dir):
            if item == ".gitkeep":
                continue
            p = snapshots_dir / item
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
            except Exception as e:
                print(f"Error removing snapshot {p}: {e}")
        print("Cleared backend/snapshots directory")

    # 4. Clear ocr-spool images directory
    if spool_images_dir.exists():
        for item in os.listdir(spool_images_dir):
            p = spool_images_dir / item
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
            except Exception as e:
                print(f"Error removing spool image {p}: {e}")
        print("Cleared backend/uploads/ocr-spool/images directory")

    # 5. Clear uploaded mp4 videos
    uploads_dir = base_dir / "uploads"
    if uploads_dir.exists():
        for item in uploads_dir.glob("*.mp4"):
            try:
                item.unlink()
            except Exception as e:
                print(f"Error removing uploaded video {item}: {e}")
        print("Cleared backend/uploads/*.mp4 videos")

    # 6. Clear logs directory
    logs_dir = base_dir / "logs"
    if logs_dir.exists():
        for item in logs_dir.glob("*.log"):
            try:
                item.unlink()
            except Exception as e:
                print(f"Error removing log {item}: {e}")
        print("Cleared backend/logs directory")

if __name__ == "__main__":
    clear_all()
