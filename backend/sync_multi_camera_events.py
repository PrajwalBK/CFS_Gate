import sys
import time
import sqlite3
from app.config.settings import Settings
from app.database.database import Database
from app.database.repositories.gate import GateRepository
from app.snapshots.snapshot_manager import SnapshotManager
from app.events.cloud_dispatcher import CloudOutboxDispatcher

def main():
    db = Database('sqlite:///data/gate.db')
    repo = GateRepository(db)
    snaps = SnapshotManager('snapshots')
    cfg = Settings(_env_file='.env')
    dispatcher = CloudOutboxDispatcher(cfg, repository=repo, snapshots=snaps)

    print("Authenticating with Prosper Cloud API...", flush=True)
    auth_headers = dispatcher.session.get_auth_headers()
    if not auth_headers.get('Authorization'):
        print("ERROR: Failed to authenticate with Prosper Cloud API.", flush=True)
        return

    print("Authentication successful.", flush=True)

    conn = sqlite3.connect('data/gate.db')
    c = conn.cursor()
    c.execute(
        "SELECT id, container_number, cloud_event_id, cloud_visit_id "
        "FROM gate_events "
        "WHERE cloud_event_id IS NOT NULL AND status = 'CONFIRMED' "
        "ORDER BY timestamp DESC"
    )
    rows = c.fetchall()
    print(f"Total confirmed events with cloud_event_id: {len(rows)}", flush=True)

    synced_count = 0
    failed_count = 0
    skipped_count = 0

    for i, (eid, cntr, cloud_id, visit_id) in enumerate(rows, 1):
        ev = repo.event(eid)
        if not ev:
            continue

        imgs, expected, captured = dispatcher._build_multi_camera_images(ev, allow_single_fallback=False)
        if len(captured) <= 1:
            skipped_count += 1
            continue

        iso_s1 = ev.get('iso_code_1') or ''
        cs1 = ev.get('container_size') or ''
        if '20' in cs1 or iso_s1.startswith('2'):
            size_1 = '20'
        elif '45' in cs1 or iso_s1.startswith('L'):
            size_1 = '45'
        else:
            size_1 = '40'

        put_payload = {
            'EventId': cloud_id,
            'VisitId': visit_id or ev.get('cloud_visit_id'),
            'Containers': [
                {
                    'ContainerNumber': cntr,
                    'ContainerNumberConfidence': 0.98,
                    'Size': size_1,
                    'ContainerSize': cs1 or None,
                    'SizeCode': iso_s1 or None,
                    'Position': 'SINGLE',
                    'Images': imgs
                }
            ]
        }

        try:
            res = dispatcher.session.put_gate_event_images(cloud_id, put_payload)
            if res and res not in ('OFFLINE', 'REJECTED'):
                uploaded_count = res.get('uploadedImagesCount', len(imgs)) if isinstance(res, dict) else len(imgs)
                cams_str = ', '.join(sorted(list(captured)))
                print(f"[{synced_count+1}] SUCCESS: {cntr} (Event {eid[:8]}) -> {uploaded_count} images ({cams_str})", flush=True)
                repo.mark_cloud_delivery(eid, 'DELIVERED', f'Multi-camera evidence updated ({len(captured)} cams)')
                synced_count += 1
            else:
                print(f"FAILED: {cntr} (Event {eid[:8]}) -> res: {res}", flush=True)
                failed_count += 1
        except Exception as exc:
            print(f"ERROR: {cntr} (Event {eid[:8]}) -> {exc}", flush=True)
            failed_count += 1

        time.sleep(0.05)

    print("\n================ Backfill Summary ================", flush=True)
    print(f"Successfully Synced : {synced_count}", flush=True)
    print(f"Failed              : {failed_count}", flush=True)
    print(f"Skipped (single-cam): {skipped_count}", flush=True)
    print("==================================================", flush=True)

if __name__ == '__main__':
    main()
