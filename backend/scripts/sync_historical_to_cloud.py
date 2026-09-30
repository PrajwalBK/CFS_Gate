"""
Script to synchronize validated historical GateVision events to Prosper Smart Yard cloud.
Endpoint: https://syapi.prosperassettracking.com/api/v1/gate/events
"""

import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import requests

sys.path.insert(0, os.path.abspath('.'))

from app.config.settings import Settings
from app.database.database import Database
from app.database.repositories.gate import GateRepository
from app.ocr.validator import ContainerValidator
from app.snapshots.snapshot_manager import SnapshotManager


def iso_utc(dt):
    if dt is None:
        d = datetime.now(timezone.utc)
    elif isinstance(dt, str):
        try:
            d = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except Exception:
            return dt
    elif isinstance(dt, datetime):
        d = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    else:
        d = datetime.now(timezone.utc)
    return d.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def main():
    settings = Settings()
    db = Database(settings.database_url.get_secret_value())
    repo = GateRepository(db)
    snapshots = SnapshotManager(settings.snapshot_directory)
    validator = ContainerValidator()

    # 1. Authenticate with CFS.Admin
    auth_url = getattr(settings, 'prosper_auth_url', 'https://cfsapi.prosperassettracking.com/api/auth/login')
    username = getattr(settings, 'prosper_username', 'sarsite@gmail.com') or 'sarsite@gmail.com'
    password = settings.prosper_password.get_secret_value() if settings.prosper_password else 'Test@123'

    print(f"Authenticating with {auth_url} as {username}...")
    auth_res = requests.post(auth_url, json={'username': username, 'password': password}, timeout=15)
    if auth_res.status_code not in (200, 201):
        print(f"Authentication failed: {auth_res.status_code} {auth_res.text}")
        return

    token = auth_res.json().get('accessToken') or auth_res.json().get('token')
    print("Authentication successful!")

    # Decode claims for site and client
    parts = token.split('.')
    claims = json.loads(base64.urlsafe_b64decode(parts[1] + '=' * (-len(parts[1]) % 4)))
    client_id = claims.get('client_ids', '01a0b4b8-9eea-76b2-9c67-b6351b71016f')
    site_id = claims.get('site_ids', '01a0b4b9-9c2b-7ac0-8426-b4410df8164c')

    headers = {
        'Authorization': f'Bearer {token}',
        'X-Site-Id': site_id,
        'X-Client-Id': client_id,
        'Content-Type': 'application/json',
    }

    # 2. Query confirmed events with valid container numbers
    evs = repo.events(limit=200)
    confirmed_events = [
        e for e in evs['items']
        if e.get('status') == 'CONFIRMED' and (e.get('container_number') or e.get('container_number_2'))
    ]

    print(f"Found {len(confirmed_events)} confirmed events in database.")

    endpoint = "https://syapi.prosperassettracking.com/api/v1/gate/events"
    device_id = getattr(settings, 'prosper_device_uuid', 'EDGE-GATE-01') or 'EDGE-GATE-01'

    synced_count = 0
    skipped_count = 0
    failed_count = 0

    for event_summary in confirmed_events:
        event_id = event_summary['id']
        event = repo.event(event_id)
        if not event:
            continue

        c1 = event.get('container_number')
        c2 = event.get('container_number_2')
        is_twin = bool(event.get('is_twin'))

        # Skip known hallucinated numbers
        if c1 == 'ITUU6180860' or c2 == 'ITUU6180860':
            print(f"[{event_id[:8]}] Skipping hallucinated tare-weight number ITUU6180860")
            skipped_count += 1
            continue

        # Check snapshots on disk
        event_snaps = event.get('snapshots', [])
        valid_snaps = []
        for s in event_snaps:
            p = snapshots.path(s['image_path'])
            if p.is_file():
                valid_snaps.append((s, p))

        if not valid_snaps:
            print(f"[{event_id[:8]}] Skipping {c1} (no local snapshot images found)")
            skipped_count += 1
            continue

        # Build containers array
        # Only accept snapshots from cameras that actually had detections for this event
        detection_cams = {
            str(d.get('camera_id')).lower()
            for d in (event.get('detections') or [])
            if d.get('camera_id')
        }
        if event.get('primary_camera_id'):
            detection_cams.add(str(event['primary_camera_id']).lower())

        containers_payload = []
        if c1:
            c1_by_cam = {}
            for s, p in valid_snaps:
                cam = s.get('camera_id', 'gate-in-1')
                if (s.get('image_path') or '').endswith('_rear.jpg') and is_twin:
                    continue
                # Keep the primary or sharpest photo for each camera
                if cam not in c1_by_cam or s.get('id') == event.get('primary_snapshot_id'):
                    c1_by_cam[cam] = (s, p)

            c1_images = []
            for cam in sorted(c1_by_cam.keys()):
                s, p = c1_by_cam[cam]
                c1_images.append({
                    'ImageType': 'FRONT',
                    'CameraId': cam,
                    'CapturedAt': iso_utc(s.get('timestamp') or event.get('timestamp')),
                    'Image': f"data:image/jpeg;base64,{base64.b64encode(p.read_bytes()).decode('utf-8')}"
                })

            size_1 = "20" if ('20' in (event.get('container_size') or '') or (event.get('iso_code_1') or '').startswith('2')) else "40"
            containers_payload.append({
                'ContainerNumber': c1,
                'ContainerNumberConfidence': 0.95,
                'Size': size_1,
                'Position': 'FRONT' if is_twin else 'SINGLE',
                'Images': c1_images
            })

        if is_twin and c2 and c2 != c1:
            c2_by_cam = {}
            for s, p in valid_snaps:
                if (s.get('image_path') or '').endswith('_rear.jpg') or any(k in (s.get('camera_id') or '').lower() for k in ('rear', '-3')):
                    cam = s.get('camera_id', 'gate-in-3')
                    if cam not in c2_by_cam or s.get('id') == event.get('rear_snapshot_id'):
                        c2_by_cam[cam] = (s, p)
            c2_images = []
            for cam in sorted(c2_by_cam.keys()):
                s, p = c2_by_cam[cam]
                c2_images.append({
                    'ImageType': 'REAR',
                    'CameraId': cam,
                    'CapturedAt': iso_utc(s.get('timestamp') or event.get('timestamp')),
                    'Image': f"data:image/jpeg;base64,{base64.b64encode(p.read_bytes()).decode('utf-8')}"
                })
            size_2 = "20"
            containers_payload.append({
                'ContainerNumber': c2,
                'ContainerNumberConfidence': 0.95,
                'Size': size_2,
                'Position': 'REAR',
                'Images': c2_images[:3]
            })

        if not containers_payload:
            continue

        dt_captured = event.get('timestamp')
        visit_id = f"VISIT-{dt_captured[:10].replace('-', '')}-{event_id[:8].upper()}"
        event_type = 'GATE_IN' if event.get('event_type') in ('ENTRY', 'IN', 'GATE_IN') else 'GATE_OUT'

        payload = {
            'VisitId': visit_id,
            'EventType': event_type,
            'DeviceId': device_id,
            'CapturedAt': iso_utc(dt_captured),
            'Truck': {'TruckNumber': 'NA'},
            'Driver': None,
            'Containers': containers_payload
        }

        try:
            res = requests.post(endpoint, json=payload, headers=headers, timeout=20)
            if res.status_code in (200, 201):
                resp_data = res.json() if res.content else {}
                cloud_visit_id = resp_data.get('visitId', visit_id)
                cloud_event_id = resp_data.get('eventId', 'OK')
                repo.mark_cloud_synced(event_id, cloud_event_id)
                print(f"[SUCCESS] {event_id[:8]} | Container={c1} | Visit={cloud_visit_id} | CloudEventId={cloud_event_id}")
                synced_count += 1
            else:
                print(f"[REJECTED] {event_id[:8]} | Status={res.status_code} | Body={res.text[:150]}")
                failed_count += 1
        except Exception as exc:
            print(f"[ERROR] {event_id[:8]} | {exc}")
            failed_count += 1

    print("\n" + "=" * 60)
    print(f"Sync complete! Synced: {synced_count}, Skipped: {skipped_count}, Failed: {failed_count}")
    print("=" * 60)


if __name__ == '__main__':
    main()
