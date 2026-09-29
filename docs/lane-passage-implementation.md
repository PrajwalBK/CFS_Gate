# Lane-based truck passages — 25 September 2026

This implementation is enabled per camera by saving one or two lane polygons
in Lane calibration. Cameras without polygons retain the existing dwell path.
Use the same lane IDs on views of the same physical road lane. Gate ID remains
separate from lane ID. Camera position/role does not determine the truck's lane.

## Processing

- The existing container/text YOLO runs alongside the local vehicle model
  (`PASSAGE_VEHICLE_MODEL_PATH`, default `models/yolov8x.pt`) on calibrated views.
  It does not download weights. The installed vehicle model contains class 7,
  truck. Both models run under the existing inference scheduling/GPU lock.
- Truck bottom-centre selects a unique road polygon. Overlapping/ambiguous lane
  membership is withheld. A tracked truck retains its lane/event as it moves.
- Text regions must belong unambiguously to a measured truck box. A separate
  candidate buffer is kept for each region, capped at three frames and six
  regions per truck. The first OCR is immediate; subsequent requests choose the
  best buffered crop, with the configured attempt and confirmation limits.
- ISO candidates are associated with the nearest number region on that truck.
  Confirming one region does not stop detection of another region on the truck.
- Independent truck events can coexist in either lane. Missing camera frames
  do not prove clearance. Finalization requires continuously measured absence
  for `GATE_CLEAR_SECONDS` and completion of outstanding OCR jobs.
- Repeated numbers count as one identity. Two voted distinct identities and
  two observed 20-foot ISO codes on a tracked chassis can become a twin.
  Missing votes, unreadable regions, mixed sizes, ambiguous ownership, or more
  than two identities produce NEEDS_REVIEW. No ISO code is invented.
- Live views with a repeated identity can combine using gate, lane, direction
  and timing. Views with different identities can also be collected into one
  candidate visit when they are from different calibrated RTSP cameras on the
  same lane and within the configured view window, so camera A may read the
  first container and camera B the second. Both views must have a validated
  identity before this candidate association is made. That association is marked
  ambiguous and remains NEEDS_REVIEW until truck identity is corroborated;
  it cannot silently become a confirmed twin. Independent uploaded videos
  remain separate because their timestamps are not a shared media clock.
- One finalized event contains the container identities, lane ID, proof crops,
  and raw context snapshots. OCR jobs retain the canonical event link. Restarted
  incomplete passages are recovered for review rather than silently published.
- Cloud selection checks identity/detection/time provenance and ignores context
  companion files. Exact duplicate front image bytes are excluded. The missing
  `utcnow` import in cloud acknowledgement persistence is fixed; queued copies
  of already acknowledged or non-confirmed events are skipped at dispatch.

## Verification and operating limits

The local vehicle model detected the two separate trucks in historical event
`d5be054c-b173-4bcb-b66b-070cc7aba6a6` at approximately 0.94 and 0.92 confidence.
This is a one-frame check, not a twin accuracy or GPU latency benchmark.

The unit/integration tests cover lane separation, preserving lane association,
bounded candidates, per-region OCR isolation, identity-based camera grouping,
rejecting independent file-video fusion, passage clearance, twin persistence,
mixed ISO review, and durable cloud acknowledgements.

Calibration must be drawn for each real camera. Generic truck boxes and IoU
tracking still need validation against site occlusion and bumper-to-bumper
traffic. A camera-only second identity without a shared truck identity is not
automatically fused. Camera-to-ground calibration or a stronger vehicle
re-identification model is needed for those ambiguous cross-camera handoffs.
The additional YOLO model increases GPU work; throughput must be measured on
the deployment device before increasing inference FPS. No instant-result or
production twin-accuracy guarantee is claimed.

No existing production events or uploaded videos were rewritten during this
implementation. The additive lane_id database column is migrated at startup.
