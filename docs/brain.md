# GateVision Gate Module — Project Brain

## Stationary queue occupancy — September 11, 2026

Each configured viewpoint has one occupancy zone (`occupancy_roi`, normalized rectangle, default full frame). Configure separate viewpoints/zones for independent lanes. Image coordinates are not compared across cameras. The existing per-frame tracker remains in place, but a stable occupancy origin now owns OCR/event identity across tracker resets.

Flow: **CLEAR → TRACKING → OCCUPIED (confirmed event) → WAITING (stationary) → CLEAR (verified departure)**. Before confirmation, OCR still uses bounded evidence samples. After confirmation, no further OCR/event writes are scheduled for that occupied view, including when it resumes moving before leaving. Already queued samples for the committed origin are completed without another OCR call or event log.

- Containers are selected from `DWELL_CONTAINER_CLASSES` (container/trailer labels by default; feet detections remain supplementary evidence). The largest eligible box establishes an empty zone's candidate.
- Normalized centroid drift below `STATIONARY_DRIFT_THRESHOLD=0.02` for `STATIONARY_SECONDS=1` means stationary. The threshold is validated below 0.05. Drift is measured against an anchor, so accumulated slow motion cannot indefinitely masquerade as tiny per-frame jitter.
- A box with IoU strictly above `STATIONARY_MATCH_IOU=0.45` reassociates to the occupancy regardless of tracker ID. A non-overlapping container in an occupied zone conservatively holds the lock; it does not claim a new identity.
- `GATE_CLEAR_SECONDS=7` of observed, container-free inference frames releases the zone. Short detection gaps reset on reappearance. Gaps longer than `GATE_OBSERVATION_GAP_SECONDS=2` restart the clearance clock: camera outages, pauses or slow inference do not count as an empty lane. Set the observation-gap threshold to suit measured inference cadence.
- Optional per-camera `gate_exit_axis: x|y`, `gate_exit_position: 0..1`, and `gate_exit_positive: true|false` release occupancy when the matched box fully crosses the calibrated departure line in the specified direction. This is separate from the existing direction-classification line. With no calibrated exit line, only clear-time release applies. Boxes already beyond the exit are excluded from new occupancy.
- Within the same view, a verified new passage is not merged into an earlier dwell event just because it falls within `DEDUP_SECONDS`. Across views of the same gate, matching confirmed container identity can bind to an existing occupied event beyond that time window; the repository verifies lane, identity and direction before binding.
- Proof selection ranks the available saved samples through confirmation using Laplacian sharpness weighted by exposure. The selected image and its detection coordinates/timestamp stay together. This is a quality heuristic over sampled evidence, not a guarantee of the globally sharpest frame. Existing definitive per-track snapshots are not overwritten by subsequent stationary frames. Active occupancy files and unconfirmed/pending evidence are protected from the existing completed-image cleanup.
- `GET /api/cameras` now includes `occupancy` state, event ID, normalized box and pending-clearance flag. No per-frame waiting log messages are emitted.

Occupancy geometry/state is in memory for the current runtime. An explicit source/calibration generation change resets it; service restarts do not yet restore the occupancy lock. Saved OCR evidence remains durable. Visually indistinguishable back-to-back containers without a clear gap or reliable exit crossing remain a conservative hold; tune the ROI/exit line using actual site video before deployment. The feature does not infer physical identity from IoU alone or guarantee behavior under prolonged occlusion. Verification: **84 passed, 1 skipped**, including ten focused occupancy/proof tests; physical-camera acceptance remains pending.

Example camera YAML additions (calibrate these values for the actual view):

```yaml
occupancy_roi: [0.1, 0.1, 0.9, 0.95]
gate_exit_axis: x
gate_exit_position: 0.9
gate_exit_positive: true
```

Last reviewed: **2026-10-01, 13:10 IST (UTC+05:30)**. For dated changes, evidence and verification results, see [Project change log](CHANGELOG.md). This document describes the current working tree, including changes not yet committed.

This document is the source of truth for how the active Gate Module is intended to work. It describes the operating model, runtime flow, interfaces, persistence, failure behavior, configuration and deployment boundaries. The former root `app/` Flask application has been retired and removed. The remaining legacy folders are not active dependencies of this module.

## 1. Purpose

GateVision observes container movement at an industrial gate through eight currently configured logical camera views. Each view can use either:

- a physical IP camera connected through the PoE network and exposed as an RTSP stream; or
- a manually uploaded video used when cameras are unavailable or for repeatable testing.

The module captures frames, detects relevant objects, tracks them across frames, reads container numbers, validates ISO 6346-style identifiers, determines movement direction when calibrated, and writes a gate event with its evidence. The dashboard also exposes input state, event history, CPU/RAM and NVIDIA GPU telemetry.

This is an observation and evidence system. It does not control a physical barrier, guarantee BIC registration, or replace a gate operator's review process.

## 2. Current operating boundaries

- The new backend is FastAPI with one in-process runtime owner.
- The frontend is a standalone Angular application.
- Production persistence is MySQL through SQLAlchemy and PyMySQL.
- SQLite is permitted only as a test double in `DEPLOYMENT_MODE=test`.
- The backend must run with exactly one worker because camera capture, tracking, queues and GPU scheduling are process-local.
- The deployment startup path is one FastAPI process. It also serves the compiled Angular files, so a separate frontend server is not required.
- Development services bind to localhost. No application authentication is implemented yet.
- Site configuration and admin login are not implemented. Current site/admin labels (including Panvel CFS) are static display text, not a verified site connection or authenticated session. The static cloud-sync label likewise does not prove successful delivery.
- Uploaded-video preview works when MySQL or AI models are unavailable.
- AI event processing requires an initialized MySQL database and successfully loaded detector/OCR models.
- Real camera accuracy, live MySQL behavior and target-edge performance must be measured on the deployment hardware.

## 3. System map

```mermaid
flowchart LR
    POE[PoE IP cameras] -->|ONVIF discovery or configured RTSP| REG[Camera source registry]
    FILE[Uploaded video files] --> LIB[Local video library]
    REG --> SLOT[Eight configurable input slots]
    LIB --> SLOT
    SLOT --> CAP[Independent capture workers]
    CAP --> LATEST[Latest-frame buffers]
    LATEST --> PREVIEW[MJPEG stream and JPEG fallback API]
    LATEST --> SCHED[Single inference scheduler]
    SCHED --> DET[YOLO detector adapter]
    DET --> TRACK[Per-slot temporal tracker]
    TRACK --> DIR[Direction logic]
    TRACK --> QUEUE[Bounded in-memory save queue]
    QUEUE --> SPOOL[(Durable crop and frame spool)]
    SPOOL --> OCR[OCR engine]
    OCR --> ISO[Container ID validator]
    ISO --> CONFIRM[Repeated-evidence confirmation]
    CONFIRM --> DEDUP[Lane/direction/time deduplication]
    DEDUP --> DB[(MySQL)]
    DEDUP --> SNAP[Snapshot files]
    PREVIEW --> API[FastAPI]
    DB --> API
    API --> UI[Angular operations dashboard]
    METRICS[CPU / RAM / GPU metrics] --> API
```

## 4. Main project areas

### Backend

- `backend/app/main.py` creates the FastAPI application, middleware, routes and lifecycle.
- `backend/app/runtime.py` coordinates inputs, model loading, inference, OCR and event creation.
- `backend/app/config/settings.py` validates environment and YAML configuration.
- `backend/app/camera/` owns RTSP/file decoding, input workers, source selection and uploaded-video assignments.
- `backend/app/detection/` contains detector and tracker boundaries.
- `backend/app/ocr/` contains cropping, OCR and container-number validation.
- `backend/app/events/` contains confirmation, direction and event-association behavior.
- `backend/app/database/` contains schema initialization, ORM models and repositories.
- `backend/app/snapshots/` owns evidence image writes.
- `backend/app/api/` contains the HTTP surface.
- `backend/app/metrics.py` and `backend/app/gpu_metrics.py` collect runtime telemetry.

### Frontend

- `frontend/src/app/app.component.ts` coordinates polling, UI actions and error isolation.
- `frontend/src/app/camera-grid.component.ts` renders configured input cards, source/position/direction dropdowns, uploads and playback controls.
- `frontend/src/app/system-health.component.ts` renders database, model, queue, CPU/RAM and GPU health.
- `frontend/src/app/event-table.component.ts` and `event-detail.component.ts` render event history and evidence.
- `frontend/src/app/gate-api.service.ts` defines typed API contracts.
- `frontend/src/styles.css` contains the responsive visual system.

### Configuration and local state

- `backend/config/cameras.yaml` defines physical source choices and eight logical processing slots.
- `backend/.env` contains local secrets and deployment-specific values; it must not be committed.
- `backend/uploads/assignments.json` stores uploaded-video assignments.
- `backend/uploads/camera-selections.json` stores the selected physical source for each slot.
- `backend/uploads/camera-roles.json` stores operator-selected position, direction and related slot configuration.
- `backend/uploads/ocr-spool/jobs.sqlite3` stores only image paths, historical job metadata, OCR results and completion state. Images are separate files under `backend/snapshots/ocr-pending/<evidence-id>/crop.png` and `frame.png`. This local job index is separate from MySQL event persistence.
- `backend/app/camera/discovery.py` handles bounded ONVIF discovery and stream negotiation.
- `backend/snapshots/` stores event evidence images.

## 5. Physical cameras and logical slots

Physical cameras and logical slots are deliberately separate concepts.

A physical camera source has:

- a stable source ID;
- a friendly name shown in dropdowns;
- an environment-variable name containing its RTSP URL, or a discovered URI held in backend memory; and
- an enabled flag.

A logical slot has:

- a slot ID and display name;
- a gate/lane ID;
- direction or line-crossing calibration;
- capture/inference rates;
- OCR crop settings; and
- an active source, which can be a physical camera or a video file.

This separation allows an operator to route any known PoE camera into any slot without sending the RTSP URL or password to the browser. The slot keeps its lane/direction/ROI semantics while only its media source changes.

### Source dropdown flow

1. The dashboard requests `GET /api/cameras/sources/available`.
2. The backend returns source ID, friendly name, configured/discovered flags and a sanitized connection hint, never a stream URL or password.
3. The operator chooses a camera in a slot dropdown.
4. The frontend sends `POST /api/cameras/{slot_id}/source` with the source ID.
5. The runtime locks that slot, stops and joins the previous worker, deactivates manual-video mode, creates a new worker for the chosen RTSP source, resets that slot's tracker and starts capture.
6. The source ID is saved in `camera-selections.json` and restored on the next service start.

`configured=true` means a configured or discovered stream URI is available. It does not prove successful streaming. The worker's ONLINE/OFFLINE/RECONNECTING/ERROR state is the connectivity result. Background discovery fills unused views; operators independently select position and direction. See section 22 for assignment and preservation rules.

## 6. Manual-video flow

Manual video is a first-class fallback for each slot.

1. The operator chooses or drops MP4, AVI, MOV, MKV, WEBM or M4V footage on a slot.
2. Angular checks the extension and configured size limit before sending.
3. The browser streams the raw file body to `POST /api/cameras/{slot_id}/video` with a URL-encoded `X-Filename` header.
4. The backend streams the body to a random internal filename and enforces the size limit while receiving it.
5. OpenCV must open the file and decode at least one frame. Empty, oversized, unsupported or unreadable files are removed and rejected.
6. A replacement worker decodes the first frame for preview and enters READY; playback does not start automatically.
7. Play starts server-side, FPS-paced decoding. Pause freezes the worker, Restart builds a fresh source generation, and Play all starts every file-backed slot.
8. Use camera restores the physical camera selected in the slot dropdown.

Uploads survive service restarts and return in READY state. Replaced or deactivated files are retained; no automatic retention/deletion policy exists. Preview uses continuous MJPEG with a JPEG fallback, without audio, and is not synchronized forensic playback across cameras.

## 7. Service startup flow

```mermaid
sequenceDiagram
    participant U as Uvicorn
    participant A as FastAPI lifecycle
    participant D as Database
    participant R as Gate runtime
    participant C as Camera workers
    participant M as AI models

    U->>A: Start one worker
    A->>D: Check schema/connectivity
    alt Database available
        A->>D: Synchronize logical camera rows
    else Database unavailable
        A->>A: Record sanitized startup error
        A->>R: Force AI processing disabled
    end
    A->>R: Start runtime
    R->>C: Restore videos or selected cameras
    R->>R: Start background ONVIF discovery and retry loop
    R->>C: Assign discovered streams to unused views
    opt PIPELINE_ENABLED=true and database ready
        R->>M: Load detector then OCR in background
        M-->>R: READY or ERROR state
    end
    A-->>U: API ready, possibly DEGRADED
```

Database failure does not terminate the API. Camera cards, dropdowns, uploads, previews and machine metrics remain usable. Event endpoints return a sanitized database-unavailable response, and the UI isolates that error from the rest of the dashboard.

## 8. Capture and frame handling

Each slot owns one `CameraWorker` thread. Workers independently open, read and reconnect so a broken source does not block the other slots.

- Live RTSP failures use bounded open/read timeouts and exponential reconnect backoff.
- File sources are paced using recorded FPS and report completion at EOF.
- Each worker holds only a latest preview and one pending inference frame.
- Replacing an unread pending frame increments the dropped-frame count instead of building latency.
- Oversized frames are aspect-ratio resized to `MAX_FRAME_WIDTH` before publication.
- Every installed source has a new random `source_id` generation.
- A frame carries slot ID, UTC timestamp, sequence, pixels and source generation.

The source generation is important: previews reject detections from replaced sources. Accepted OCR evidence keeps its original source generation, timestamp, lane and role; durable historical jobs are processed even after a source switch or restart, never relabeled as the replacement input.

## 9. AI processing flow

### Scheduling

There is one inference scheduler, a bounded in-memory save queue, a dedicated disk writer and one OCR consumer of durable jobs. The accelerator lock serializes GPU OCR with YOLO; CPU OCR does not hold it. Saving first improves evidence durability, not GPU throughput.

For each eligible frame:

1. Enforce the slot/global inference FPS and frame-skip rules.
2. Run the detector once.
3. Retain recent detections in a bounded in-memory list for diagnostics.
4. Update that slot's class-aware IoU tracker.
5. Wait for the minimum track hit count.
6. Determine ENTRY, EXIT or UNKNOWN.
7. If the tracked object class is OCR-eligible, enqueue an evidence sample subject to the per-track interval, sample limit (`OCR_MAX_ATTEMPTS`) and save-queue capacity. The writer commits crop, original frame and metadata before OCR can consume it.

### Detector

The `Detector` interface isolates model implementation. The current Ultralytics adapter accepts explicit PyTorch `.pt`, ONNX `.onnx` or TensorRT `.engine` configuration. The service never silently downloads a replacement checkpoint or substitutes synthetic results.

Model class names must be verified against the actual checkpoint. Aliases are configuration, not evidence that similarly named training labels mean the same physical object.

### Tracking and direction

The current tracker is a proof-of-concept, class-aware IoU tracker with time-based expiry. It is local to each view and does not perform cross-camera person/object re-identification.

Direction is one of:

- fixed ENTRY or EXIT for a calibrated unidirectional view;
- a result from crossing a configured normalized x/y line and deadband; or
- UNKNOWN when direction is not safely determined.

Camera position names do not imply direction. Line placement and positive-crossing orientation must be calibrated with site footage.

### OCR and ISO validation

OCR operates on a configurable normalized sub-region inside a detected container/trailer box, with the whole box as the default. The crop is clipped to image bounds before reading.

The OCR factory supports EasyOCR and the configured oLmOCR/Qwen2-VL adapter. Model packages, weights and available GPU memory must match the chosen engine.

### ISO 6346 Size-Type Code & Feet Extraction

Standard ISO 6346 container markings specify length, height, and general-purpose dry freight type using a 4-character code (e.g. `45G1`, `22G1`, `42G1`, `L5G1`).
- **Correction of 'C' to 'G'**: OCR engines frequently confuse 'G' with 'C' (e.g., `45C1` or `22C1`). Because ISO 6346 contains no container type 'C', all pattern matches `[1-4LMN][0-9]C[0-9A-Z]` are deterministically corrected to `G` (`45G1`, `22G1`, etc.).
- **Feet Size Parsing**: Validated codes are translated into standardized feet sizes (`45G1` -> `40 FT HC`, `42G1` -> `40 FT`, `22G1` -> `20 FT`, `L5G1` -> `45 FT HC`, direct feet text `40FT` -> `40 FT`) and given the validation status `VALID_SIZE_CODE`.
- **Event Presentation**: Container events present both the normalized container number and the parsed feet size badge in the UI table and event detail view.

Validation preserves raw OCR text and creates a normalized candidate. A trusted container ID requires:

- three owner-code letters;
- equipment category U, J or Z;
- six serial digits;
- one check digit; and
- a valid calculated check digit (with check-digit-directed repair for G/C substitution in owner prefixes).

Ambiguous multiple identifiers and guessed O/0 replacements are rejected. Valid text must be independently confirmed across the configured number of observations. Exhausted invalid/insufficient reads create NEEDS_REVIEW evidence without a trusted container number.

## 10. Event creation and deduplication

An OCR job produces an observation containing the origin key, lane, track, detection, validation, direction, image and timing.

The origin key includes service run, slot, source generation and track ID. SQL uniqueness makes retries idempotent. A track is considered complete only after its database transaction succeeds.

For a confirmed container identity, the repository looks for an event with the same:

- gate/lane;
- normalized container number;
- direction; and
- timestamp inside `DEDUP_SECONDS`.

If found, the new camera/track evidence is associated with that event. Otherwise a new event is created. Different container identities are never combined merely because they occurred close together. UNKNOWN and known directions remain distinct.

Invalid reads stay local to their track and create NEEDS_REVIEW events; they are not safely fused across views. Repeated real visits inside the dedup window can merge, while fragmented tracks outside it can duplicate. Tune this using site acceptance data.

## 11. Atomic evidence persistence

Schema version 1 contains:

| Table | Purpose |
|---|---|
| `gate_schema_version` | Explicit schema compatibility marker |
| `cameras` | Logical slot metadata and sanitized configuration |
| `gate_events` | Lane-level movement event and confirmed/review status |
| `detections` | Per-source track evidence, class, confidence and bounding box |
| `ocr_results` | Raw/normalized text, confidence and validation flags |
| `snapshots` | Database reference to evidence JPEG |
| `system_logs` | Structured event audit records without source credentials |

Event, detection, OCR, snapshot metadata and audit row are committed in one SQL transaction. A snapshot is first written through a temporary path and renamed. If the SQL transaction fails, the new JPEG is removed. A process crash between filesystem write and SQL commit can still leave an orphan file, so maintenance must reconcile snapshot files with the database.

Committed OCR evidence and validation results survive restarts in the local spool. Failed OCR/event writes retry from disk; successful event writes are idempotent by origin key. The bounded capture-to-save staging queue is still volatile: a crash before the spool transaction commits can lose those not-yet-saved samples. See section 25 for capacity and recovery details.

Optional Prosper cloud delivery uses `CloudOutboxDispatcher`, a bounded **in-memory** queue with retry/backoff for events and evidence uploads. Despite its name, this is not a durable database outbox. Enable it with reviewed `PROSPER_*` configuration; cloud authentication is separate from dashboard login. Verify actual delivery and secure the endpoint before deployment.

## 12. API contract

### Inputs

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/cameras` | All logical slots, active source and worker/playback state |
| GET | `/api/cameras/sources/available` | Sanitized source dropdown choices |
| GET | `/api/cameras/discovery/status` | Latest scan state, count and safe message |
| POST | `/api/cameras/discovery/rescan` | Request an earlier discovery scan |
| POST | `/api/cameras/{id}/role` | Save position and ENTRY/EXIT/UNKNOWN direction |
| POST | `/api/cameras/{id}/source` | Select a physical camera for a slot |
| GET | `/api/cameras/{id}/status` | One slot's worker state |
| POST | `/api/cameras/{id}/test` | Attempt one frame from the active source |
| GET | `/api/cameras/{id}/frame` | Latest resized JPEG preview with no-store caching |
| GET | `/api/cameras/{id}/stream` | Continuous MJPEG preview that follows source replacement |
| POST | `/api/cameras/{id}/video` | Stream and install an uploaded video |
| POST | `/api/cameras/{id}/playback` | Play, pause, restart or restore-camera |
| POST | `/api/videos/play-all` | Start all installed videos |
| POST | `/api/processing/start` | Recheck MySQL and begin AI model loading |

### Events and evidence

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/gate-events` | Paginated events with exact container, camera, type and status filters |
| GET | `/api/gate-events/{id}` | Event with detection, OCR and snapshot evidence |
| GET | `/api/detections` | Recent transient or stored detection view, as implemented |
| GET | `/api/ocr-results` | Recent stored OCR results |
| GET | `/api/snapshots/{id}` | Evidence image by database ID |

### Operations

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | READY or DEGRADED system state |
| GET | `/api/models` | Detector/OCR loading and error state plus configured backend |
| GET | `/api/metrics` | FPS, counters, latency, queues, CPU/RAM/process memory and GPU readings |

Camera APIs expose sanitized source metadata, not RTSP URLs or credentials. Cloud integration error/log handling needs a separate security review before deployment; do not assume all remote errors are safe to log.

## 13. Health and telemetry

The UI polls operational data, including discovery status, every 2.5 seconds. Camera previews use persistent MJPEG streams; a one-second refresh token remains available for the JPEG fallback.

Health reports:

- database READY/UNAVAILABLE;
- pipeline enabled/disabled;
- detector and OCR state;
- configured/online slot counts;
- model/processing/startup error type; and
- upload-size limit.

Metrics report:

- capture FPS per worker;
- inference FPS per slot;
- dropped/skipped/error/event counters;
- inference/OCR/end-to-end latency samples with p50/p95;
- OCR queue depth;
- CPU, system memory and process memory; and
- GPU name, utilization, power draw, power limit, temperature and VRAM when supported.

NVIDIA GPU values come from a bounded, cached `nvidia-smi` query. Unsupported fields remain null and the UI shows unavailable rather than inventing a reading. Jetson deployments without compatible `nvidia-smi` need a platform metrics adapter such as a reviewed tegrastats integration.

## 14. Dashboard behavior

The dashboard has eight configured independent camera cards, discovery status with Search again, system health, and overview/events/event-detail navigation.

Each camera card shows:

- logical view number, name, lane and direction;
- ONLINE/OFFLINE/RECONNECTING/ERROR or file playback state;
- camera-source dropdown;
- position and IN/OUT/Unknown direction dropdowns, persisted across restarts;
- live or recorded latest-frame preview;
- drag/drop and Add/Replace video control;
- upload progress and validation errors;
- per-video Play/Pause/Restart/Use camera controls; and
- capture FPS, inference FPS and dropped frames.

The system panel shows database/model states, OCR queue, CPU/RAM and GPU telemetry. The event area supports exact container search, movement filter, pagination and evidence details. Database event errors are displayed locally and do not hide input or health controls.

Responsive styles adapt camera cards and health panels to the viewport; the camera list is driven by backend slot configuration.

## 15. Configuration reference

Core settings are environment driven:

- `DATABASE_URL`: new MySQL database connection.
- `CAMERAS_FILE`: path to YAML source/slot definitions.
- `CAMERA_1_RTSP` … `CAMERA_8_RTSP`: manual RTSP secret variables.
- `CAMERA_DISCOVERY_ENABLED`, `CAMERA_DISCOVERY_INTERVAL_SECONDS`: startup discovery and retry interval.
- `CAMERA_ONVIF_USERNAME`, `CAMERA_ONVIF_PASSWORD`: shared camera ONVIF credentials.
- `PROSPER_*`: optional cloud endpoint, credentials and delivery settings.
- `PIPELINE_ENABLED`: enable AI loading at startup.
- `MODEL_BACKEND`, `MODEL_PATH`, `MODEL_DEVICE`, `MODEL_IMAGE_SIZE`.
- `TARGET_CLASSES`, `CLASS_ALIASES`, `OCR_CLASSES`.
- `OCR_ENGINE`, `OCR_GPU`, `OCR_DOWNLOAD_ENABLED`, `OCR_MODEL_DIRECTORY`.
- `INFERENCE_FPS`, `CONFIDENCE_THRESHOLD`, `OCR_MIN_CONFIDENCE`.
- `MIN_TRACK_HITS`, `OCR_CONFIRMATIONS`, `OCR_MAX_ATTEMPTS`, `OCR_INTERVAL_SECONDS`.
- `OCR_QUEUE_SIZE` (memory save staging), `OCR_SPOOL_MAX_MB` (retained evidence capacity), `DEDUP_SECONDS`, `TRACK_TTL_SECONDS`.
- `SNAPSHOT_DIRECTORY`, `UPLOAD_DIRECTORY`, `MAX_UPLOAD_MB`.
- `CORS_ORIGINS` and capture open/read/reconnect/frame-width limits.

RTSP credentials belong only in `.env` or the deployment's secret store. YAML refers to the environment-variable name. Never place live passwords in frontend code, API payloads, logs or committed configuration.

## 16. Failure behavior

| Failure | Expected behavior |
|---|---|
| Camera URL missing | Source shown as not configured; slot remains OFFLINE |
| Camera unreachable/read failure | Worker reports error/reconnecting with exponential backoff; other slots continue |
| No discovery response | Display network/setup guidance and retry; preserve current inputs |
| Discovered camera authentication fails | List camera with setup warning; do not auto-assign it |
| Source switched | Old worker stops; tracker resets; saved OCR evidence retains its historical identity |
| Invalid upload | Request rejected and partial file removed; previous active source remains intact |
| Video EOF | Last preview remains; state becomes COMPLETED; Restart is available |
| MySQL unavailable at startup | API runs DEGRADED; AI disabled; inputs/uploads/previews/metrics remain available |
| MySQL write fails during event | Saved OCR job remains pending and retries; health/metrics expose failure |
| Detector/OCR load failure | Model state becomes ERROR; API and video operation continue |
| Save staging queue full | New unsaved sample is dropped and `ocr_queue_dropped` increments; capture remains current |
| Spool full or disk write fails | Writer retries the current sample; health exposes `ocr_spool_error`; no existing evidence is deleted |
| GPU telemetry unsupported | GPU fields are null and UI displays unavailable |
| Worker fails to stop in time | Source replacement returns conflict; operator retries after shutdown |

## 17. Security and operational controls

- Backend and frontend default to loopback development access.
- CORS permits configured frontend origins.
- Mutating browser requests with an unexpected `Origin` are rejected.
- Origin checks are not user authentication or authorization.
- Production exposure requires an authenticated TLS reverse proxy, user roles, network policy and audit requirements.
- Uploaded filenames are sanitized; server filenames are random; traversal outside the upload root is rejected.
- Upload extension, length and actual decoder readability are validated.
- API responses and logs do not include RTSP URLs.
- MySQL schema initialization is explicit and does not drop legacy tables.
- No automatic snapshot/upload retention currently runs; disk capacity and removal policy are operator responsibilities.

## 18. Development and verification

### Normal Windows startup

From the project root:

```powershell
.\start.ps1
```

This one command starts the UI, API, input workers and AI runtime at `http://127.0.0.1:8001`. It builds Angular only if the compiled UI is missing. Use `./start.ps1 -BuildFrontend` after frontend changes. Binding to `0.0.0.0` is available through `-HostAddress 0.0.0.0`, but requires deployment authentication, TLS and network controls before untrusted exposure.

### Component development

From `backend`:

```powershell
.\.venv\Scripts\python -m app.database.initialize
.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 1 --no-access-log
.\.venv\Scripts\python -m pytest -q
```

From `frontend`:

```powershell
npm ci
npm start
npm run build
```

The Angular development proxy forwards `/api` to `127.0.0.1:8001`. Open `http://127.0.0.1:4200`. OpenAPI is available at `http://127.0.0.1:8001/docs`.

`python -m tools.simulate --serve` provides explicitly marked synthetic events for hardware-free UI smoke testing. It is not real AI output. Recorded-video tests use actual OpenCV decoding. The MySQL integration suite requires a disposable `gate_test_<suffix>` database through `TEST_MYSQL_URL` and otherwise skips.

## 19. Site acceptance checklist

Before production acceptance, verify:

1. Every PoE camera's IP, RTSP path, credentials, codec and network reachability.
2. Every dropdown source maps to the intended physical camera.
3. Eight simultaneous streams over a sustained run, including discovery, unplug/reconnect, IP changes and saved-role restoration.
4. Lane IDs and ENTRY/EXIT calibration against actual vehicle movement.
5. Detector classes against the exact deployed checkpoint metadata.
6. OCR visibility across daylight, darkness, rain, glare, dirt, angle and motion blur.
7. Exact container-number rate and NEEDS_REVIEW rate on unseen site footage.
8. Missed-event, duplicate-event and cross-view association rates.
9. CPU, GPU, memory, queue depth, dropped frames, disk growth and p50/p95 latency.
10. MySQL backup/restore, service supervision, power-loss behavior and restart recovery.
11. Snapshot/upload retention and orphan-file reconciliation.
12. Authentication, authorization, TLS, network segmentation and audit policy before non-loopback exposure.

## 20. Known limitations and next hardening phases

- No login, role-based access, site-management workflow or multi-site tenancy yet.
- Discovery supports directly attached private IPv4 networks and ONVIF Media1. It does not provision camera networks/accounts or guarantee discovery across VLANs or proprietary protocols.
- The IoU tracker is not sufficient for all long occlusions or complex traffic.
- Cross-camera association relies on confirmed container identity plus lane/direction/time, not visual re-identification.
- Local OCR evidence is durable after its spool commit. Pre-save staging and the Prosper dispatcher remain memory-only; cloud delivery is not a durable outbox.
- No automatic upload/snapshot lifecycle management exists.
- No synchronized multi-camera replay clock or recorded-timestamp extraction exists.
- Preview is MJPEG with a JPEG fallback, not WebRTC, and does not carry audio.
- No dedicated container-text localization model is included.
- TensorRT/Jetson packages, engine compatibility and throughput must be validated on the target device.
- Real model accuracy and performance are not established by unit tests.
- Schema changes require reviewed versioned migrations; the initializer is not a migration engine.

Recommended progression:

1. Connect and calibrate physical cameras and collect representative site footage.
2. Validate/iterate the detector and OCR pipeline using held-out data.
3. Run disposable MySQL integration and sustained eight-stream soak tests.
4. Add durable outbox, retention/reconciliation jobs and operational alerting.
5. Add authenticated site/admin management before network exposure.
6. Optimize decode/inference for the selected edge hardware only after measurement.

## 21. Definition of a successful gate event

A confirmed production event is successful when:

- the correct physical source was active in the intended logical lane slot;
- a relevant object was detected and tracked with sufficient evidence;
- direction was correctly calibrated or honestly remained UNKNOWN;
- OCR produced a non-ambiguous, check-digit-valid container number with repeated confirmation;
- deduplication created or associated the correct lane passage;
- detection, OCR and snapshot evidence committed together;
- the event is visible and reviewable in the dashboard; and
- no credential, fabricated identity or stale-source evidence entered the record.

When these conditions cannot be met, the system should fail visibly and conservatively—OFFLINE, DEGRADED, ERROR or NEEDS_REVIEW—while preserving independent operation where it is safe to do so.

## 22. Automatic camera discovery (September 10 update)

The current dashboard has eight configurable views and continuous MJPEG previews. The September 9 event-detail pages, OCR options, and in-memory cloud dispatcher remain in place.

Start the complete application from the project root with `./start.ps1`. Startup launches discovery in the background: send a bounded ONVIF WS-Discovery multicast probe on directly attached private IPv4 interfaces, request each responding camera's Media service and first media profile, then request its RTSP stream URI. Another scan runs 60 seconds after completion; **Search again** requests an earlier scan. Discovery failures do not prevent the dashboard from starting.

Reachable cameras appear in the source dropdowns. Cameras with an available stream automatically fill unused enabled views, in discovery order. Saved operator selections, configured RTSP inputs, and active uploaded videos are preserved. Newly auto-filled views use **Unassigned** position and **Unknown** direction: the operator chooses Front/Top, Left, Right, or Rear, and IN (ENTRY) or OUT (EXIT). Additional discovered cameras remain selectable after all eight views are occupied; inventory is bounded to 32 discovered devices. Discovery does not infer physical position.

Source selections and role assignments are saved under `UPLOAD_DIRECTORY` in `camera-selections.json` and `camera-roles.json`. Stable ONVIF device IDs allow selections to reconnect after a restart or IP change once discovery succeeds again. Role changes reset the view's tracking/calibration (full-image OCR region, no crossing line), invalidate queued observations from the old role, and assign ENTRY/EXIT views to lane-in/lane-out. Recalibrate as needed after changing a role. Unknown views remain separate unassigned lanes.

### Camera/network setup

- Connect the host and PoE cameras to a reachable camera network; enable ONVIF discovery and RTSP on each camera. The software discovers the cameras, not the PoE switch itself.
- In `backend/.env`, set `CAMERA_ONVIF_USERNAME` and `CAMERA_ONVIF_PASSWORD` to a shared camera ONVIF account if authentication is required. No passwords are guessed. Restart after changing credentials.
- `CAMERA_DISCOVERY_ENABLED=true` is the default. `CAMERA_DISCOVERY_INTERVAL_SECONDS=60` controls the retry interval (15–3600 seconds).
- Devices requiring credentials still appear with a setup warning; they are not automatically assigned until a stream URI can be obtained. Different per-camera accounts can use the existing manual `CAMERA_N_RTSP` configuration.
- VLAN isolation, firewalls blocking UDP multicast port 3702, disabled ONVIF, Media2-only cameras, IPv6-only cameras, or proprietary cameras can prevent automatic discovery. Use manual RTSP configuration for unsupported devices. RTSP availability still depends on camera permissions and codec support.
- Discovered RTSP credentials stay in backend memory; the dashboard receives only IDs, names and safe connection hints. Discovery does not follow endpoint redirects or accept camera-advertised service/stream endpoints on another host.

Protocol reference: [ONVIF Media service operations](https://www.onvif.org/yaml/wsdl-viewer.php?file=%2Fver10%2Fmedia%2Fwsdl%2Fmedia.wsdl). Hardware interoperability and full-load eight-camera operation require on-site validation; automated tests are not hardware certification.

## 23. Latest verification snapshot

The September 10 implementation was checked with backend pytest (**58 passed, 1 skipped**) and a successful Angular production build. A bounded local ONVIF probe found **0 cameras**. Live-camera interoperability, eight-camera throughput and production cloud delivery were not verified by those checks. This documentation-only update did not rerun application tests. See the change log for timestamp provenance.

## 24. Low-latency preview and OCR ordering (September 10, 14:50 IST)

This update supersedes the dashboard MJPEG description above. Each camera card now uses `/api/cameras/{id}/live`, a same-origin/allowed-origin WebSocket. The browser requests another JPEG only after displaying the previous one, with a 66 ms interval (up to approximately 15 FPS before network/encoding overhead). It keeps one frame in flight, avoiding an old-frame queue and the browser's HTTP/1.x connection limit for eight persistent MJPEG requests. MJPEG and single-JPEG endpoints remain available for other clients. Angular's development proxy enables WebSocket forwarding; deployment proxies must also support WebSocket upgrades. Install the new `websockets` requirement before startup.

Capture and YOLO rates are independent. Increasing preview FPS does not increase detection throughput. YOLO processes the latest pending frame only when the accelerator is available. CPU OCR (`OCR_GPU=false`) no longer holds the accelerator lock; GPU OCR still serializes with YOLO to avoid concurrent GPU inference. A long Qwen GPU call can therefore still interrupt detection. CPU OCR or a separately provisioned OCR accelerator is needed to avoid that particular shared-GPU bottleneck; neither guarantees zero end-to-end latency. Existing camera FPS and model settings are not silently increased.

Preview boxes use only the latest detection result from the matching source generation, expire after 300 ms relative to the preview frame, and clear when YOLO returns no detections. This prevents trails of old boxes; it is not motion interpolation or frame-perfect synchronization. OpenCV buffer-size tuning is best-effort after opening, not an unsupported mandatory FFmpeg open parameter.

**OCR ordering:** superseded by the implemented save-first workflow in section 25. Uploaded source videos are still decoded into memory; selected crops and original frames are now committed to the spool before OCR reads them.

Reference: [MDN HTTP/1.x connection management](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Connection_management_in_HTTP_1.x). Actual camera-to-screen latency and detection FPS must be measured with all eight physical cameras and the deployment GPU; local timestamps do not measure camera-side encoder buffering.

## 25. Durable save-first OCR (September 10 implementation)

Current flow: **capture → YOLO/tracking → bounded save staging → save crop/full-frame PNG files → commit file paths/job metadata → OCR → persist validation → confirmation/event storage → mark sample DONE**.

- `OCRJobStore` writes lossless PNG files under `SNAPSHOT_DIRECTORY/ocr-pending/<evidence-id>/`: `crop.png` and `frame.png`. Each file is flushed to disk and renamed from a temporary file before its job is committed to `UPLOAD_DIRECTORY/ocr-spool/jobs.sqlite3`. SQLite contains paths/metadata/status, not image bytes, and uses `synchronous=FULL`. It is not a replacement for the MySQL event database. No pickle data or stream credentials are stored.
- At startup, the earlier BLOB-based job store is migrated automatically: export files first, change the metadata schema in one transaction, preserve job IDs/results/retries, then reclaim old database pages. Failed exports roll back database changes; already exported matching files are reusable on retry. Stop the old application before starting the updated version; do not run mixed versions against the same store.
- Metadata includes original camera ID, source generation, role, lane, direction, timestamp, bounding box, sequence, track and related detections. Source switches/restarts do not discard or reassign accepted evidence. Later live frames from other views are not attached to historical jobs.
- Saving runs on its own thread. Once a sample is saved, tracking may submit another at `OCR_INTERVAL_SECONDS`, up to `OCR_MAX_ATTEMPTS` samples per track. These are interval-selected eligible detections, not a new image-sharpness ranking algorithm.
- Pending jobs are discovered from disk when the OCR worker starts. The normal model/database prerequisites still apply. Saved validation is reused after interrupted event writes; completed samples supply confirmation votes after restart. The existing unique origin key prevents duplicate event evidence if a crash occurs between the event commit and spool completion.
- Failed jobs retain images and retry after five seconds; other eligible jobs can proceed. A completed sample can be awaiting more confirmation votes without having an event yet. Short tracks with too few samples retain their images/results but do not automatically produce a confirmed event.
- `OCR_SPOOL_MAX_MB=10240` limits the registered image-file/metadata payload (filesystem/SQLite overhead and orphan files can use additional space). Both pending and completed evidence count. Completed files stay in their original `ocr-pending` folder; SQLite status, not the folder name, indicates completion. There is no automatic deletion or retention job. Back up both image folders and the job index. Do not move/delete image files while jobs reference them.
- A crash between file saves and job registration can leave unregistered image files. They are preserved, not deleted; deterministic evidence IDs allow a retry of the same job to reuse matching files. This is a recoverable file-first workflow, not a single atomic filesystem/database transaction.
- Health exposes `ocr_spool` counts and `ocr_spool_error`. Metrics expose durable `ocr_queue_depth`, memory `ocr_save_queue_depth`, saved-job/error counters and spool totals. A full staging queue increments `ocr_queue_dropped`; no zero-loss guarantee is made before disk commit, during hardware failure or when storage capacity is exhausted.
- GPU OCR still competes with YOLO on the same accelerator. This change protects saved evidence; it does not guarantee no missed detections or zero live-stream latency.
- Verification for this update: **68 backend tests passed, 1 skipped**, including simulated interruption and disk-write failure recovery. This is not a physical power-loss or eight-camera throughput test.

### Folder-storage update

The September 10 folder-storage revision supersedes the original BLOB implementation. The migration and file-first path are tested for rollback, partial writes, pixel fidelity and restart recovery: **71 passed, 1 skipped** in the full backend suite. Images are now browsable with ordinary image viewers under the snapshots folder; the database contains no image columns of BLOB type after migration.

## 26. Multi-instance launcher and dynamic port conflict resolution (October 1, 2026)

To support co-located deployments and concurrent testing environments across different directory trees (e.g. primary repository and secondary instances on different drives), the project launcher scripts have been upgraded with non-destructive, dynamic port conflict resolution:

- **Launch Scripts**: Root `start.bat` provides a single-click entry point on Windows that executes `start.ps1` with appropriate ExecutionPolicy bypass.
- **Port Conflict Probing**: `start.ps1`, `setup_and_run.bat`, and `setup_and_run.ps1` probe target port `8001` using socket availability checks.
- **Non-Destructive Allocation**: If port `8001` is already in use by another instance or process, the launcher does NOT terminate the existing process (all `Stop-Process` and `taskkill /F` logic on port 8001 has been removed). Instead, the launcher scans subsequent ports (`8002` through `8010`) to identify the first available port and binds `uvicorn` and the Angular frontend proxy to that port.
- **Script Syntax Resilience**: Corrected batch execution parsing in `setup_and_run.bat` by eliminating unescaped parentheses within parenthesized conditional blocks.

## 27. OCR offline resilience and fallback hardening (October 1, 2026)

- **PaddleX Model Source Check Suppression**: Set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` in environment variables within `backend/app/ocr/ocr_engine.py` prior to model instantiation. This prevents PaddleOCR from initiating external network queries on startup, ensuring predictable and fast initialization in air-gapped or restricted-connectivity industrial networks.
- **Graceful Fallback Protection**: Hardened `FallbackOCREngine` initialization to intercept `ImportError` and `ModuleNotFoundError` (e.g., when optional heavy dependencies such as `transformers` or `torchvision` are absent in environments configured for `OCR_FALLBACK_ENGINE=olmocr`). This safeguards the persistent spool worker from crashing with unhandled exceptions and prevents cascading `saved_ocr_failed` errors.
- **Lane Calibration & Role Defaulting**: Camera slot `gate-in-3` role is assigned to `REAR` to ensure proper check-digit extraction on entry vehicles, and `occupancy_roi` default in `lane_calibration.json` is set to full frame `[0.0, 0.0, 1.0, 1.0]` to prevent aggressive cropping from filtering out approaching trucks.

## 28. Delimiter-tolerant OCR validation and fast-passage confirmation (October 1, 2026)

- **Delimiter-Tolerant ISO 6346 Parsing**: Real-world container markings often contain vertical lines, dashes, or spacing between the 4-letter owner prefix and the 6-digit serial number (`TRHU | 154589 [5]`). OCR models frequently transcribe delimiters as digits (e.g. `TRHU1 1545895`). The validator now detects separated 8-digit blocks, tests checksum validity upon delimiter detachment, and successfully verifies valid ISO 6346 identifiers (`TRHU1545895`).
- **Multiline Code Reconstruction**: Container codes stamped across multiple lines on doors or corner posts (`CSQU\n305438\n3`) are unified into single-line candidate streams.
- **Removed Size Marking Fallback Hijacking**: Fixed a regression where presence of standard container size codes (`22G1`, `45G1`) in raw text caused the validator to truncate the normalized container text to `'22G'` and force `VALID_SIZE_CODE` (`NEEDS_REVIEW`), discarding the valid container number.
- **Fast-Passage Confirmation**: When tracking completes and the attempt ceiling is reached, if at least one sample yielded a valid check-digit read without conflicting candidates, `EventManager` confirms the event rather than discarding the valid read into `NEEDS_REVIEW`.
- **Windows MAX_PATH Protection**: Shortened temporary file generation in `OCRJobStore` to prevent path overflows on deep directory structures.
## 29. Dual-application coexistence and snapshot publishing architecture (October 1, 2026)

- **Port Isolation & UI Proxy Targeting**:
  - When running in environments where another instance or project occupies port `8001` (such as a parallel development directory on another disk drive), `frontend/proxy.conf.json` is aligned to target `http://127.0.0.1:8002`.
  - Default listening port for `start.ps1` and `backend/start.ps1` is defaulted to `8002`, eliminating port collision warnings while maintaining automatic incrementation to subsequent ports (`8003+`) if needed.
  - `CORS_ORIGINS` in `backend/.env` is expanded to permit `http://localhost:4201`, `http://127.0.0.1:4201`, and `http://127.0.0.1:8002`.
- **Snapshot Directory Junction Automation**:
  - Automatically initializes a Windows NTFS directory junction from `snapshots` at the repository root to `backend/snapshots`, ensuring that users inspecting either the root folder or backend folder immediately see all generated event snapshots and proof crops.
- **Occupancy Sensitivity & Unconfirmed Snapshot Preservation**:
  - Decreased the minimum candidate bounding box threshold in `ViewOccupancy` from `0.015` to `0.005`, accommodating wide-angle camera perspectives and distant lane approaches without missing valid container passages.
  - Streamlined `EventManager.observe`: eliminated redundant attempt-count gates so that once a passage completes its physical entrance (`entrance_complete=True`), unconfirmed events (`status: NEEDS_REVIEW`) are saved to the database and multi-camera proof snapshots are written to disk.
