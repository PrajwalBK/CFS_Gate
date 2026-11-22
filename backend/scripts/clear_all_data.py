import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

def clear_all():
    base_dir = Path(__file__).resolve().parents[1]
    parent_dir = base_dir.parent.parent
    db_path = base_dir / "data" / "gate.db"
    spool_db_path = base_dir / "uploads" / "ocr-spool" / "jobs.sqlite3"
    snapshots_dir = base_dir / "snapshots"
    evidence_dir = base_dir / "evidence"
    spool_images_dir = base_dir / "uploads" / "ocr-spool" / "images"

    # 1. Clear gate.db tables (preserving cameras & schema)
    if db_path.exists():
        try:
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
        except Exception as e:
            print(f"Error clearing gate.db: {e}")

    # Remove SQLite WAL & SHM files if present
    for extra in [base_dir / "data" / "gate.db-wal", base_dir / "data" / "gate.db-shm"]:
        if extra.exists():
            try:
                extra.unlink()
            except Exception as e:
                print(f"Could not remove {extra.name}: {e}")

    # 2. Clear OCR spool database
    if spool_db_path.exists():
        try:
            conn = sqlite3.connect(spool_db_path)
            cur = conn.cursor()
            cur.execute("DELETE FROM jobs")
            conn.commit()
            cur.execute("VACUUM")
            conn.commit()
            conn.close()
            print("Cleared ocr-spool jobs.sqlite3")
        except Exception as e:
            print(f"Error clearing spool db: {e}")

    # 3. Fast-swap & recreate snapshots directory (handles millions of files instantly)
    to_delete_dirs = []
    if snapshots_dir.exists():
        snap_temp = base_dir / f"snapshots_old_{int(time.time())}"
        try:
            snapshots_dir.rename(snap_temp)
            to_delete_dirs.append(snap_temp)
        except Exception:
            # Fallback if rename fails
            for item in snapshots_dir.iterdir():
                if item.name == ".gitkeep":
                    continue
                try:
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
                except Exception:
                    pass
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    (snapshots_dir / ".gitkeep").touch(exist_ok=True)
    print("Reset backend/snapshots directory to fresh empty state")

    # 3b. Fast-swap & recreate evidence directory
    if evidence_dir.exists():
        ev_temp = base_dir / f"evidence_old_{int(time.time())}"
        try:
            evidence_dir.rename(ev_temp)
            to_delete_dirs.append(ev_temp)
        except Exception:
            for item in evidence_dir.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
                except Exception:
                    pass
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "GATE_IN").mkdir(exist_ok=True)
    (evidence_dir / "GATE_OUT").mkdir(exist_ok=True)
    print("Reset backend/evidence directory to fresh empty state")

    # 4. Clear ocr-spool images directory
    if spool_images_dir.exists():
        try:
            shutil.rmtree(spool_images_dir, ignore_errors=True)
        except Exception:
            pass
    spool_images_dir.mkdir(parents=True, exist_ok=True)
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
                with open(item, "w", encoding="utf-8") as f:
                    f.truncate(0)
            except Exception:
                pass
        print("Cleared backend/logs directory")

    # 7. Clear live temporary frame images
    for live_img in base_dir.glob("live_*.jpg"):
        try:
            live_img.unlink()
        except Exception:
            pass

    # 8. Clear root workspace evidence, snapshots, logs, tmp, tmp_snaps
    for root_target in ["evidence", "snapshots", "logs", "tmp", "tmp_snaps"]:
        r_path = parent_dir / root_target
        if r_path.exists():
            for item in r_path.iterdir():
                if item.name == ".gitkeep":
                    continue
                try:
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
                except Exception:
                    pass
            print(f"Cleared workspace root/{root_target}")

    # Remove temporary swapped directories asynchronously in Windows
    for td in to_delete_dirs:
        try:
            subprocess.Popen(f'cmd /c "rd /s /q "{td}""', shell=True)
            print(f"Background deletion spawned for {td.name}")
        except Exception as e:
            print(f"Error spawning background deletion for {td}: {e}")

    print("\n>>> Application successfully restored to completely fresh state! <<<")

if __name__ == "__main__":
    clear_all()
