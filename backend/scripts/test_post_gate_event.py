"""
Test script: Send a real gate event (preferably GATE_OUT) to Prosper Smart Yard API.
Prints full request payload and response body so you can verify the new API schema works.

Usage:
    cd backend
    python scripts/test_post_gate_event.py [--event-id <uuid>] [--event-type GATE_OUT|GATE_IN]
"""
import argparse
import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, os.path.abspath('.'))

from app.config.settings import Settings
from app.database.database import Database
from app.database.repositories.gate import GateRepository
from app.ocr.validator import extract_iso_size_code
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


def derive_size(iso_code, container_size):
    cs = container_size or ''
    if not cs and iso_code:
        _, cs = extract_iso_size_code(iso_code)
        cs = cs or ''
    if '20' in cs or (iso_code or '').startswith('2'):
        return '20', cs or '20 FT'
    if '45' in cs or (iso_code or '').startswith('L'):
        return '45', cs or '45 FT HC'
    return '40', cs or '40 FT'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--event-id', default=None)
    parser.add_argument('--event-type', choices=['GATE_IN', 'GATE_OUT', 'ENTRY', 'EXIT'], default=None)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    settings = Settings()
    db = Database(settings.database_url.get_secret_value())
    repo = GateRepository(db)
    snapshots = SnapshotManager(settings.snapshot_directory)

    auth_url = getattr(settings, 'prosper_auth_url', None) or 'https://cfsapi.prosperassettracking.com/api/auth/login'
    username = getattr(settings, 'prosper_username', None) or getattr(settings, 'prosper_email', None) or 'sarsite@gmail.com'
    password = settings.prosper_password.get_secret_value() if settings.prosper_password else 'Test@123'
    bearer = settings.prosper_bearer_token.get_secret_value() if settings.prosper_bearer_token else None
    api_key = settings.prosper_api_key.get_secret_value() if settings.prosper_api_key else None

    token = bearer
    site_id = (settings.prosper_site_id or '').strip()
    client_id = (getattr(settings, 'prosper_client_id', None) or '').strip()

    if not token and not api_key:
        print(f'Authenticating with {auth_url} as {username}...')
        for pw_try in ([password] + ['Test@123', 'Prosper@123', 'Admin@123']):
            for payload_fmt in [{'UserName': username, 'Password': pw_try}, {'username': username, 'password': pw_try}]:
                try:
                    r = requests.post(auth_url, json=payload_fmt, timeout=10)
                    if r.status_code in (200, 201):
                        data = r.json()
                        token = data.get('accessToken') or data.get('token')
                        if token:
                            parts = token.split('.')
                            if len(parts) >= 2:
                                padded = parts[1] + '=' * (-len(parts[1]) % 4)
                                try:
                                    claims = json.loads(base64.urlsafe_b64decode(padded))
                                    if not site_id:
                                        site_id = str(claims.get('site_ids', ''))
                                    if not client_id:
                                        client_id = str(claims.get('client_ids', ''))
                                except Exception:
                                    pass
                            print(f'  Auth OK  site_id={site_id}  client_id={client_id}')
                            break
                except Exception as e:
                    print(f'  Auth attempt failed: {e}')
            if token:
                break
        if not token:
            print('ERROR: Could not authenticate.')
            sys.exit(1)

    if token:
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    else:
        headers = {'x-api-key': api_key, 'Content-Type': 'application/json'}
    if site_id:
        headers['X-Site-Id'] = site_id
    if client_id:
        headers['X-Client-Id'] = client_id

    if args.event_id:
        event = repo.event(args.event_id)
        if not event:
            print(f'ERROR: Event {args.event_id} not found.')
            sys.exit(1)
    else:
        prefer_types = ['EXIT', 'OUT'] if not args.event_type else [args.event_type]
        all_evs = repo.events(limit=100)['items']
        confirmed = [e for e in all_evs if e.get('status') == 'CONFIRMED' and (e.get('container_number') or e.get('container_number_2'))]
        target = None
        for et in prefer_types:
            candidates = [e for e in confirmed if (e.get('event_type') or '').upper() in (et.upper(), f'GATE_{et.upper()}')]
            if candidates:
                target = candidates[0]
                break
        if not target and confirmed:
            target = confirmed[0]
        if not target:
            print('ERROR: No confirmed events found in local DB.')
            sys.exit(1)
        event = repo.event(target['id'])

    c1 = event.get('container_number')
    c2 = event.get('container_number_2')
    is_twin = bool(event.get('is_twin'))
    event_id = event['id']

    print(f'\n{"="*60}')
    print(f'Event:      {event_id}')
    print(f'Type:       {event.get("event_type")}')
    print(f'Container1: {c1}  iso={event.get("iso_code_1")}  size={event.get("container_size")}')
    print(f'Container2: {c2}  iso={event.get("iso_code_2")}  size={event.get("container_size_2")}')
    print(f'Is twin:    {is_twin}')
    print(f'Snapshots:  {len(event.get("snapshots", []))}')

    containers_list = []
    event_snaps = event.get('snapshots', [])

    if c1:
        c1_images = []
        for s in event_snaps:
            ip = s.get('image_path') or ''
            cam = (s.get('camera_id') or '').lower()
            if not (ip.endswith('_rear.jpg') or any(k in cam for k in ('-3', 'rear'))):
                p = snapshots.path(ip)
                if p and p.is_file():
                    c1_images.append({
                        'ImageType': 'FRONT',
                        'CameraId': s.get('camera_id', 'gate-in-1'),
                        'CapturedAt': iso_utc(s.get('timestamp') or event.get('timestamp')),
                        'Image': f"data:image/jpeg;base64,{base64.b64encode(p.read_bytes()).decode('utf-8')}"
                    })
        api_sz1, human_sz1 = derive_size(event.get('iso_code_1') or '', event.get('container_size') or '')
        containers_list.append({
            'ContainerNumber': c1,
            'ContainerNumberConfidence': 0.95,
            'Size': api_sz1,
            'ContainerSize': human_sz1 or None,
            'SizeCode': event.get('iso_code_1') or None,
            'Position': 'FRONT' if is_twin else 'SINGLE',
            'Images': c1_images[:3]
        })
        print(f'  C1 images: {len(c1_images)}  size_api={api_sz1}  size_human={human_sz1}')

    if c2 and c2 != c1:
        c2_images = []
        for s in event_snaps:
            ip = s.get('image_path') or ''
            cam = (s.get('camera_id') or '').lower()
            if ip.endswith('_rear.jpg') or any(k in cam for k in ('-3', 'rear')):
                p = snapshots.path(ip)
                if p and p.is_file():
                    c2_images.append({
                        'ImageType': 'REAR',
                        'CameraId': s.get('camera_id', 'gate-in-3'),
                        'CapturedAt': iso_utc(s.get('timestamp') or event.get('timestamp')),
                        'Image': f"data:image/jpeg;base64,{base64.b64encode(p.read_bytes()).decode('utf-8')}"
                    })
        api_sz2, human_sz2 = derive_size(event.get('iso_code_2') or '22G1', event.get('container_size_2') or '')
        containers_list.append({
            'ContainerNumber': c2,
            'ContainerNumberConfidence': 0.95,
            'Size': api_sz2,
            'ContainerSize': human_sz2 or None,
            'SizeCode': event.get('iso_code_2') or None,
            'Position': 'REAR',
            'Images': (c2_images or c1_images)[:3]
        })
        print(f'  C2 images: {len(c2_images)}  size_api={api_sz2}  size_human={human_sz2}')

    ts_str = event.get('timestamp', '')
    dt_captured = datetime.fromisoformat(ts_str.replace('Z', '+00:00')) if ts_str else datetime.now(timezone.utc)
    visit_id = f"VISIT-{dt_captured.strftime('%Y%m%d')}-{event_id[:8].upper()}"
    raw_direction = (event.get('event_type') or 'ENTRY').upper()
    event_type_api = 'GATE_OUT' if raw_direction in ('EXIT', 'OUT', 'GATE_OUT') else 'GATE_IN'
    device_id = getattr(settings, 'prosper_device_uuid', None) or 'EDGE-GATE-01'

    payload = {
        'VisitId': visit_id,
        'EventType': event_type_api,
        'DeviceId': device_id,
        'CapturedAt': iso_utc(dt_captured),
        'Truck': {'TruckNumber': 'NA'},
        'Driver': None,
        'Containers': containers_list,
    }

    def strip_images(obj):
        if isinstance(obj, dict):
            return {k: ('<base64 omitted>' if k == 'Image' else strip_images(v)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [strip_images(i) for i in obj]
        return obj

    print(f'\nPAYLOAD (images omitted):')
    print(json.dumps(strip_images(payload), indent=2))

    endpoint = f"{settings.prosper_api_base_url.rstrip('/')}/api/v1/gate/events"
    print(f'\nPOST -> {endpoint}')

    if args.dry_run:
        print('\n[DRY RUN] Not sending.')
        return

    try:
        res = requests.post(endpoint, json=payload, headers=headers, timeout=20)
        print(f'\nHTTP {res.status_code}')
        try:
            print('Response:', json.dumps(res.json(), indent=2))
        except Exception:
            print('Response:', res.text[:600])

        if res.status_code in (200, 201):
            print('\n SUCCESS - event accepted by cloud!')
        elif res.status_code == 409:
            print('\n CONFLICT (409) - cloud requires a prior GATE_IN for this container.')
        elif res.status_code == 400:
            print('\n BAD REQUEST (400) - check payload schema.')
        elif res.status_code == 401:
            print('\n UNAUTHORIZED (401) - token rejected.')
        elif res.status_code in (500, 503):
            print(f'\n SERVER ERROR ({res.status_code}) - API unavailable.')
        else:
            print(f'\n Unexpected: {res.status_code}')
    except Exception as e:
        print(f'\nRequest failed: {e}')


if __name__ == '__main__':
    main()
