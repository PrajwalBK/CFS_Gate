import json
import sys
import time

sys.path.insert(0, '.')
from app.camera.lens_control import DahuaLensClient

cameras = [
    ('gate-in-1', '192.168.1.204'),
    ('gate-in-2', '192.168.1.203'),
    ('gate-in-3', '192.168.1.205'),
    ('gate-out-1', '192.168.1.207'),
    ('gate-out-2', '192.168.1.208'),
    ('gate-out-3', '192.168.1.209'),
]

results = {}

for name, host in cameras:
    print(f"\n==================== Tuning {name} ({host}) ====================")
    client = DahuaLensClient(host=host, username='admin', password='Prosper@123')
    if not client.login():
        print(f"FAILED to login to {name} ({host})")
        results[name] = "LOGIN_FAILED"
        continue

    # 1. VideoInDayNight: Set Auto mode across all profiles
    dn_res = client._rpc('/cpapi2', {
        "method": "configManager.getConfig",
        "params": {"name": "VideoInDayNight[0]"},
        "id": 10,
        "session": client.session
    })
    dn_table = dn_res.get('params', {}).get('table', [])
    for p in dn_table:
        p['Mode'] = 'Auto'
        p['Sensitivity'] = 2
        p['Delay'] = 6
    set_dn = client._rpc('/cpapi2', {
        "method": "configManager.setConfig",
        "params": {"name": "VideoInDayNight[0]", "table": dn_table},
        "id": 11,
        "session": client.session
    })
    dn_ok = bool(set_dn.get('result'))
    print(f"  DayNight -> Auto: {'OK' if dn_ok else 'FAILED'}")

    # 2. VideoInBacklight: Enable GlareInhibition (HLC) to eliminate nighttime headlight glare
    bl_res = client._rpc('/cpapi2', {
        "method": "configManager.getConfig",
        "params": {"name": "VideoInBacklight[0]"},
        "id": 20,
        "session": client.session
    })
    bl_table = bl_res.get('params', {}).get('table', [])
    for p in bl_table:
        p['Mode'] = 'GlareInhibition'
        p['GlareInhibition'] = 60
    set_bl = client._rpc('/cpapi2', {
        "method": "configManager.setConfig",
        "params": {"name": "VideoInBacklight[0]", "table": bl_table},
        "id": 21,
        "session": client.session
    })
    bl_ok = bool(set_bl.get('result'))
    print(f"  Backlight -> GlareInhibition (HLC 60): {'OK' if bl_ok else 'FAILED'}")

    # 3. VideoInExposure: Ensure Auto mode (Mode 8), max shutter 8ms (1/125s) to eliminate motion blur, GainMax 85
    exp_res = client._rpc('/cpapi2', {
        "method": "configManager.getConfig",
        "params": {"name": "VideoInExposure[0]"},
        "id": 30,
        "session": client.session
    })
    exp_table = exp_res.get('params', {}).get('table', [])
    for p in exp_table:
        p['Mode'] = 8  # Customized Auto mode
        p['Value1'] = 0.1  # Min shutter: 0.1ms (1/10000s) for bright sunny days
        p['Value2'] = 8.0  # Max shutter: 8.0ms (~1/125s) to prevent motion blur at night
        p['GainMax'] = 85  # Boost low light sensitivity up to 85
    set_exp = client._rpc('/cpapi2', {
        "method": "configManager.setConfig",
        "params": {"name": "VideoInExposure[0]", "table": exp_table},
        "id": 31,
        "session": client.session
    })
    exp_ok = bool(set_exp.get('result'))
    print(f"  Exposure -> Mode 8, MaxShutter 8ms, GainMax 85: {'OK' if exp_ok else 'FAILED'}")

    # 4. Lighting: Ensure Auto lighting mode
    lt_res = client._rpc('/cpapi2', {
        "method": "configManager.getConfig",
        "params": {"name": "Lighting[0]"},
        "id": 40,
        "session": client.session
    })
    lt_table = lt_res.get('params', {}).get('table', [])
    for p in lt_table:
        p['Mode'] = 'Auto'
        p['Correction'] = 50
    set_lt = client._rpc('/cpapi2', {
        "method": "configManager.setConfig",
        "params": {"name": "Lighting[0]", "table": lt_table},
        "id": 41,
        "session": client.session
    })
    lt_ok = bool(set_lt.get('result'))
    print(f"  Lighting -> Auto: {'OK' if lt_ok else 'FAILED'}")

    results[name] = {
        'DayNight': dn_ok,
        'Backlight_HLC': bl_ok,
        'Exposure': exp_ok,
        'Lighting': lt_ok
    }

print("\n==================== SUMMARY ====================")
print(json.dumps(results, indent=2))
