# Gate Module backend

This is the active FastAPI/MySQL Gate service. The former root `../app` Flask application has been retired and removed; the remaining `../web` folder is not used by this service. Run commands **from this backend directory** so Python imports this service's `app` package. The complete operating design is in [the project brain](../docs/brain.md).

## Development startup

Use Python 3.11+ and a virtual environment. Install the right PyTorch distribution before AI dependencies; Jetson requires wheels/packages matching its JetPack. CPU development does not need TensorRT, CUDA or WebRTC.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

On Linux, activate `.venv/bin/activate` and use `python` for the following commands. Configure `.env` with a **new MySQL database** and a user authorized to create its tables. The database must already exist; schema initialization creates tables, indexes, keys and version metadata only. Do not point this at the legacy application database or another application's existing tables.

If Docker is available, the included Compose file provisions MySQL on local port **3307** to avoid an existing local server on 3306. Set `GATE_MYSQL_PASSWORD` and `GATE_MYSQL_ROOT_PASSWORD` in `.env`, run `docker compose up -d mysql`, and set `DATABASE_URL` to `mysql+pymysql://gate:<URL-encoded-password>@127.0.0.1:3307/gate_module`.

```powershell
.\.venv\Scripts\python -m app.database.initialize
.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 1 --no-access-log
```

Open `http://127.0.0.1:8001/docs` for OpenAPI. **Exactly one worker/process** owns cameras, GPU scheduling and event writing. A missing schema leaves the API available with degraded health and AI processing stopped. Uploads and video previews still work; initialize/fix the DB, then use **Enable AI**.

For normal Windows startup from the project root, use `./start.ps1`. It serves the compiled Angular UI and all APIs from one FastAPI process on port 8001. `./start.ps1 -BuildFrontend` forces a fresh UI build. The separate Angular development server is needed only while editing the frontend.

## Test without connected cameras

Start the backend and Angular frontend; MySQL and AI models are not required just to preview footage. In each of the four camera cards, choose **Add video** or drop a file. Supported extensions are MP4, AVI, MOV, MKV, WEBM and M4V; the installed OpenCV decoder must support the file's codec. The default limit is 2048 MB per file (`MAX_UPLOAD_MB`).

Each upload becomes READY with a first-frame preview. Use its **Play**, **Pause** and **Restart** controls independently, or **Play all videos**. Playback is server-decoded video shown through refreshed JPEG previews, not an audio player. Play-all does not guarantee frame synchronization between recordings. AI detection and OCR run only after models and MySQL are configured and enabled.

Video assignments persist in `uploads/assignments.json` and return to READY after restart. **Use camera** restores the camera currently selected for that slot. Replacing a video or restoring a camera retains uploaded clips on disk; storage retention is operator-managed. Uploads stay on this backend computer. Keep this unauthenticated development service on loopback; origin checks are not authentication. Site/admin setup is deliberately deferred.

Camera dropdown choices come from `camera_sources`, while the selected source per slot is saved in `uploads/camera-selections.json`. Selecting a camera immediately stops the previous source in that slot, deactivates manual-video mode, resets its tracker and begins the chosen RTSP stream. The API and UI expose only source IDs, friendly names and whether an environment variable is populated—not URLs or credentials. “Configured” does not guarantee connectivity; the card's ONLINE/OFFLINE state reflects the worker connection.

## Enable AI and cameras

1. Add each physical PoE camera under `camera_sources` in `config/cameras.yaml`, using a friendly ID/name and the name of its environment variable. Put the corresponding RTSP URLs in `.env` (`CAMERA_1_RTSP` through `CAMERA_4_RTSP` are included as examples). The four entries under `cameras` are logical viewing/processing slots with their own lane, direction, rate and ROI settings. A source may be selected for any slot from its dashboard dropdown. Blank sources are labelled “not configured” and report OFFLINE.
2. Install `requirements-ai.txt`. Choose `MODEL_PATH`, `MODEL_BACKEND` and `MODEL_DEVICE` explicitly. The existing checkpoint can be selected at `../runs/detect/gate_detector/weights/best.pt`. Verify its actual label metadata; examples of aliases are not proof that a class is a whole container. Remove incorrect aliases instead of relabeling physical objects blindly.
3. Set `OCR_ENGINE=paddleocr` for the production reader. On Windows RTX 50-series systems, `requirements-ai.txt` installs Paddle 3.3 from Paddle's CUDA 13 index; its PP-OCRv6 models are cached locally on first startup. Paddle runs in a dedicated GPU worker because Windows cannot safely load Paddle CUDA 13 and PyTorch CUDA 12 into the same process. Set `OCR_FALLBACK_ENGINE=olmocr` to use Qwen only when PP-OCR returns no usable container or ISO text. Set `OCR_GPU=true` and `MODEL_DEVICE=cuda`.
4. Set `PIPELINE_ENABLED=true`, restart and inspect `/api/models` and `/api/health`. Failed model loading is reported without terminating the API. The normal service never substitutes mock detections or downloads a replacement YOLO checkpoint.

Capture drains live feeds independently and publishes at `capture_fps`. A single latest-frame slot avoids accumulating latency. Inference is capped by `inference_fps` (per-camera override) and optional `frame_skip`. Capture FPS and inference FPS are separately measured. Large input frames are scaled to `MAX_FRAME_WIDTH` (default 1920), so verify OCR legibility before reducing it for performance. Bounding boxes refer to those processed original-aspect frames.

`TARGET_CLASSES` and `CLASS_ALIASES` configure physical objects. Only `OCR_CLASSES` (container/trailer by default) drive container-ID events; truck-only detections are visible in the recent detection API. The detector, tracker and OCR are separate interfaces/modules. Ultralytics supplies fit/pad preprocessing. `ocr_roi: [x1,y1,x2,y2]` selects a normalized region inside a detected box; whole-box OCR is the default until site views are calibrated. No dedicated container-text-region model is shipped.

ISO validation retains raw text and a normalized candidate, checks three owner letters + U/J/Z + six serial digits + check digit, and rejects ambiguous multiple IDs. It does not silently replace O/0 or assert BIC registration. Confirmation requires repeated valid reads above confidence threshold. Failed/insufficient reads produce NEEDS_REVIEW events after the configured attempts, without a trusted container number. OCR engine failures are reported and retried on subsequent eligible observations; no fake text is stored.

## Direction and deduplication

Leave `direction: UNKNOWN` unless a fixed view is known to observe a unidirectional lane. For a bidirectional view, configure `line_axis: x` or `y`, normalized `line_position`, `line_deadband`, and `positive_crossing: ENTRY` or `EXIT`. The tracked center must move across both sides before that camera produces a crossing event. Calibrate the line against real footage. Names such as “front” do not imply entry.

An origin key combines service run, camera and track. It is unique in SQL and a track is marked complete only after a successful transaction. Validated identities within one lane, direction and `DEDUP_SECONDS` window associate into one event with additional camera evidence. During a dwell/queue passage, the same validated container is additionally protected by `SAME_CONTAINER_EVENT_WINDOW_SECONDS` (default 15 minutes), so tracker resets and camera handoffs cannot create duplicate visits. Different identities never merge by time alone. Identity queries survive service restarts. Invalid reads remain local to their track and cannot be safely fused across viewpoints.

The POC IoU tracker is class-aware but does not guarantee identity through long occlusions, sharp viewpoint changes or detector dropout. A repeated visit inside the dedup window may merge; a fragmented track outside it may duplicate. Tune the window using site footage and replace the tracker/association module if gate traffic requires stronger passage boundaries. Opposite/unknown directions stay separate. This system does not actuate barriers.

## Persistence and snapshots

Schema version 1 has `cameras`, `gate_events`, `detections`, `ocr_results`, `snapshots`, `system_logs`, and `gate_schema_version`. SQLAlchemy supports MySQL through PyMySQL. All timestamps are UTC, and API timestamps explicitly include `Z`. Event evidence is relational; recent transient detections are a bounded 300-item memory buffer. The service does not store every frame or upload/delete legacy SQLite records.

One JPEG context image is saved for each new contributing camera/track evidence. Event, detection, OCR, snapshot metadata and audit row commit together; file writes use a temporary filename followed by rename. Rollbacks remove the newly written JPEG. A process crash between filesystem write and SQL commit may leave an orphan JPEG; reconcile unreferenced files against `snapshots.image_path` during maintenance. Retention is operator-managed for this POC; no automatic evidence deletion is enabled. Allocate/monitor disk accordingly.

Database write failures retain the current OCR job in memory for retry while the service is running. Remaining OCR/frame queues are bounded and report drops. A process/power failure can lose uncommitted jobs; a durable local outbox is a later hardening phase. Shutdown signals capture workers and drains no new work; remaining uncommitted buffers may be discarded. Capture open/read timeouts bound normal RTSP waits. Drivers/model inference that ignore timeouts may require supervisor process restart; a join timeout is logged.

Schema initialization is an explicit v1 bootstrap, not an automatic schema migrator. A different schema version is refused. Introduce a reviewed migration script before changing deployed table definitions. Backup policy, authenticated reverse proxy/TLS, service supervision and site network exposure are deployment responsibilities. Development binds to loopback and contains no user-auth implementation.

## APIs / frontend

- `GET /api/cameras`, `GET /api/cameras/{id}/status`: sanitized configuration and health.
- `GET /api/cameras/sources/available`, `POST /api/cameras/{id}/source`: list safe dropdown choices and select a source for a logical slot.
- `POST /api/cameras/{id}/test`: test an already configured source.
- `GET /api/cameras/{id}/frame`: latest JPEG, no disk write or WebRTC dependency.
- `POST /api/cameras/{id}/video`: raw video body with URL-encoded `X-Filename`; streamed size limit and decoder validation.
- `POST /api/cameras/{id}/playback`: JSON `action` of `play`, `pause`, `restart` or `restore-camera`.
- `POST /api/videos/play-all`, `POST /api/processing/start`: start file playback or enable configured AI processing.
- `GET /api/gate-events`: `limit`, `offset`, exact `container`, `camera_id`, `event_type`, `status` filters.
- `GET /api/gate-events/{id}`: related detection, OCR and snapshot evidence.
- `GET /api/ocr-results`, `GET /api/detections`, `GET /api/snapshots/{id}`.
- `GET /api/health`: 200 READY or 503 DEGRADED; disabled AI is explicitly visible.
- `GET /api/models`, `GET /api/metrics`: loading/error state, measured FPS, latency p50/p95, queue drops, CPU/memory, and GPU name/utilization/power in watts/temperature/VRAM through `nvidia-smi`. Measurements have a two-second cache and subprocess timeout. Unsupported readings are null, never fabricated (Jetson needs a platform metrics adapter).

Camera configuration changes are made in YAML/environment with restart. Stream credentials are never returned by these APIs or included in structured application logs. Native OpenCV decoder verbosity is suppressed to avoid URL leakage. The existing legacy config/UI still contains its earlier credentials and is not migrated automatically.

Start the Angular app in `../frontend`: `npm ci`, then `npm start`. Its dev proxy forwards `/api` to port 8001. `npm run build` produces `dist/gate/browser` for a static server with an `/api` reverse proxy. Angular polls current data every 2.5 seconds and refreshes previews every second. It provides independent camera/video controls, system/GPU telemetry, paginated event filters, OCR confidence, detection class/confidence and event snapshots. Event-storage failure does not suppress available camera or telemetry data. Legacy WebRTC can be integrated separately later.

## Tests and simulation

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m tools.simulate
.\.venv\Scripts\python -m tools.simulate --serve
```

The deterministic simulation writes two synthetic gate events from four views to a fresh temporary SQLite database. Images are explicitly marked SYNTHETIC TEST FIXTURE. It exists for API/UI smoke tests and is never used as live AI output. Real operation always uses MySQL. Unit tests use SQLite as a fast relational test double and include actual OpenCV recorded-video decoding.

For actual footage, set `CAMERAS_FILE=config/replay.example.yaml` and `VIDEO_1_PATH` in `.env`; enable the real detector/OCR and use MySQL. Files are paced at their FPS and report `file_completed` at EOF. Set `loop_file: true` only for soak tests. Multiple files started together use playback-time UTC timestamps; this is not a synchronized forensic replay of their original recording clocks.

For MySQL integration, create a disposable database named `gate_test_<suffix>` and set `TEST_MYSQL_URL` to its full connection URL before running `tests/test_mysql.py`. The integration test removes only rows associated with its random test lane. It is skipped when no URL is supplied; MySQL DDL compilation still runs in the standard suite.

## Edge deployment and acceptance

Set `DEPLOYMENT_MODE=edge`, configure RTSP sources, and use the same domain/API code. Select `.pt`, `.onnx` or `.engine` through the detector boundary. Install compatible ONNX Runtime or TensorRT separately. Export/validate TensorRT on the target Jetson with its actual JetPack; no binary compatibility or FPS is assumed. The current capture adapter uses portable OpenCV/FFmpeg; integrate and measure a target-specific GStreamer/NVDEC adapter if CPU decode is limiting.

Before site acceptance: validate four simultaneous camera streams, disconnection/reconnection, lighting and OCR visibility, unseen-clip accuracy, false positives, missed/duplicate passages, crossing directions, CPU/GPU/memory/disk behavior and end-to-end latency under sustained traffic. Confirm engine class metadata and train/validation/test separation. Live RTSP, real OCR accuracy, target MySQL access, long-duration durability and Jetson acceleration require measurements in that environment.
