# Production validation — 9 October 2026

Update after the snapshots archive was supplied: see [the real-data archive validation](snapshot-archive-validation/README.md). All 973 event exports and 4,313 proof references were audited, 54 saved frames were replayed in both normal and reversed arrival order, and isolated HTTP delivery verified original camera-frame bytes. The final regression result is 206 passed / 3 skipped; production acceptance remains blocked by detector misses, OCR gaps and unestablished source synchronization. The original notes below describe the earlier automated-only run.

## Scope and outcome

The full collected backend suite and frontend production build were run. Tests use isolated databases, synthetic camera frames, mocked model outputs, and a real localhost HTTP recording server. No fabricated event was sent to the production cloud service.

Latest regression results after local-media findings: **200 backend tests passed, 3 skipped**, in 42.61 seconds; **frontend production build passed**. The skips cover unavailable real-OCR reference footage, unconfigured MySQL integration, and opt-in live deployment checks. One Starlette TestClient deprecation warning remains. `git diff --check` reported no whitespace errors.

This is automated regression validation, not production acceptance or a guarantee that OCR never misreads a real container. Camera synchronization, model accuracy, lighting, occlusion, lane calibration, and sustained production throughput require representative synchronized recordings with labelled container identities.

## Changes verified

- ISO 6346 checksum enforcement at consensus, persistence, and delivery boundaries.
- Persistent delivery outcomes distinguish acknowledged delivery from offline, conflict, rejection, blocked evidence, and duplicates.
- Finalized passages and selected frames survive persistence failures and restart through a durable outbox; replay is idempotent.
- OCR executes from the durable spool in order. Finalization waits for queued evidence to be saved and pending OCR to finish.
- Delayed OCR retains its original event and slot, including after restart. Missing slots no longer fall back to another active container.
- Tracker identifiers are scoped to their camera; differing identifiers on other cameras do not split the passage.
- Different checksum-valid container identities in one passage require review instead of silently replacing the accepted identity.
- Camera frame candidates require a physical container detection from that exact camera and timestamp. Text-only, truck-only, and borrowed detections are excluded.
- Targeted OCR excludes another container's text outside the selected physical container box and excludes detections from other captures.
- Each saved observation/session image records its event, camera, capture time, source generation, sequence, container identity, physical detection, and SHA-256 digest.
- Canonical image selection excludes missing, changed, weak, or unverifiable evidence. Cloud delivery cannot replace the missing recorded image with a similarly named file. Snapshot retrieval returns 404 when the exact recorded file is missing.
- Production delivery rebuilds payloads from recorded evidence even when a caller supplies a payload. A recorded camera with unverifiable evidence blocks delivery.
- Outbound confidence uses the recorded OCR value; unknown size remains unknown rather than defaulting to 40 feet.

## Coverage

The collected suite covers API routes, event storage, camera configuration/discovery, file/video ingestion, reconnect behavior, preview, detector contracts, temporal tracking, direction/occupancy logic, passages, twin 20-foot workflows, multi-camera frame selection, OCR crops and validation, consensus, spool durability, cloud authentication/status/retries, deployment database checks, alerts/metrics, and adversarial identity boundaries.

The localhost receiver test asserts the actual JSON received over HTTP, including container number, size, confidence, camera identity, and the exact bytes decoded from the transmitted base64 image. The evidence test independently compares stored JPEGs with the selected input frames and rejects image replacement or deletion.

## Remaining acceptance work

- Real PaddleOCR acceptance requires the reference image; its test now reports a skip when absent instead of silently passing.
- MySQL integration needs TEST_MYSQL_URL. SQLite tests do not establish MySQL deployment behavior.
- Live deployment checks are read-only and opt-in with `--run-live-deployment`; they were not run against a production service.
- Frontend compilation passed; browser-based operator workflows were not exhaustively exercised.
- OCR engines and detector tests largely use test doubles. There is no measured real-camera precision/recall or proof that every camera sees the same physical container in all traffic conditions.
- Cross-camera association still depends on configured gate/slot timing and tracking. It does not independently read and verify the number in every camera image.
- No sustained multi-camera soak/load test, network partition exercise against the real server, or staged release was performed.
- Older stored images without the new provenance will be blocked for automatic cloud delivery and need operator review or newly verified capture evidence.

Before production acceptance, replay synchronized labelled camera recordings for consecutive containers, twins, unreadable markings, night/rain/glare, occlusions, tracker resets, and out-of-order arrivals against a staging API, then compare every accepted event and camera image with ground truth. The recordings were subsequently located in `D:/videos` and downloaded archives. Real local-model evaluation and replay results are in [local-media-validation/README.md](local-media-validation/README.md). A synchronized event/camera ground-truth manifest and a staging URL distinct from production have not been established.


The later real-media audit exposed unsafe character synthesis in both validator and consensus; those paths were removed. It also verified that the installed detector checkpoint has very poor physical-container recall on the available labelled test set. The stronger checkpoint is prepared separately and has not replaced production weights. The earlier automated results alone must not be used as production acceptance.
