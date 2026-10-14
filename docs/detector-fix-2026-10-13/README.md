# Container detector correction — 13 October 2026

**Inactive:** the user requested the earlier detector. Local configuration has been restored to `models/best.pt`, 640 pixels and confidence 0.28. The evaluation below describes the reverted experiment.

The local application now selects `backend/models/acceptance-candidate.pt` instead of the poorly localizing `best.pt`. Final inference settings are **640 pixels**, **confidence 0.35** and **empty class aliases**. The old weights remain intact. No running service was restarted and no production API events were sent.

Checkpoint SHA-256: `ab133ac4314ad164a5f76ed4a0cb5ddfb61f0d2d67c4d7becc95b772756fc9e2`.

## Cause and changes

The active checkpoint frequently detected number markings without an accurate physical-body box. Expanding those boxes geometrically would invent physical evidence, so this fix selects the independently evaluated local checkpoint instead.

Changed files:

- `backend/app/config/settings.py`: default checkpoint selection.
- `backend/.env`: local checkpoint selection, explicit 640-pixel inference and confidence changed from 0.28 to 0.35; camera URLs, credentials and other values preserved.
- `backend/.env.example`: same selection/settings; remove the incorrect example mapping from `Cointainer` to physical `container`; align target/OCR classes with the application.
- `backend/tests/test_detection.py`: regression proving `Cointainer` remains a number marking, `Feet` remains ISO text and only `container_object` becomes the physical body.
- `backend/README.md`: activation, model provisioning and class semantics.

The checkpoint's existing binary was not retrained, rewritten or downloaded. It is ignored by Git and must be provisioned separately on another machine. The usual application restart loads the new settings. For model-only rollback, choose `MODEL_PATH=models/best.pt`, `MODEL_IMAGE_SIZE=640` and `CONFIDENCE_THRESHOLD=0.28`; the old model's detection limitations will return.

## Actual model verification

The evaluation uses the application's real `YoloDetector`, `D:/Cointainer.v1i.yolov8/test` (873 images), same-class one-to-one box matching at IoU >= 0.5 and the dataset's body/number/ISO labels. This is fixed-threshold precision/recall, not mAP or OCR accuracy. Earlier source-stem checks found no training/test filename-stem overlap, but cannot establish absence of all scene similarity.

| Configuration | Correct body detections | False body detections | Missed bodies |
| --- | ---: | ---: | ---: |
| Previous best.pt at 640 / 0.28, earlier local evaluation | 9 | see earlier report | 864 |
| Replacement at 640 / 0.35, fresh full evaluation | 873 | 0 | 0 |
| Replacement at 960 / 0.35, fresh full evaluation, rejected | 797 | 711 | 76 |

At the final 640 setting, number-box precision/recall was 96.56%/97.80%; ISO-box precision/recall was 99.27%/99.71%. See `candidate-heldout/detector-metrics.json`.

The 960 experiment helped a four-image composite but degraded the complete held-out evaluation substantially. It was **not retained**. `candidate-heldout-960` and `active-configuration-archive.json` document that rejected experiment; the latter's name denotes the temporarily configured experiment, not the final active configuration.

Fresh comparison on the 54 archived camera frames: previous checkpoint found a body in 4 frames; replacement at final 640 settings found a body in 40 frames. Daylight groups improved from 0/14 and 0/3 to 11/14 and 3/3; night/glare improved from 4/37 to 26/37. These are body-presence counts, not independently labelled per-frame localization accuracy. See `archive-model-comparison.json`.

Warm 640 inference in that isolated comparison was approximately 31 ms median / 47 ms p95 for the replacement. This excludes decoding, tracking, OCR, queueing and API acknowledgment; it is not a three-second passage-to-server benchmark.

## Limits

- The final 640 model correctly localized both bodies in a two-image composite. A four-image panorama only localized two of four bodies correctly; this fixture combines independent photographs, not an actual simultaneous trailer scene. Wide crowded scenes still require validation and possibly a separately evaluated model/tiling strategy.
- The artificial cartoon-like dummy image still had a partial box (IoU about 0.377). The real labelled dataset result is stronger evidence for the actual camera domain; a synthetic cartoon is not a production acceptance source.
- Night/glare misses remain in the archive. The replacement substantially improves the main checkpoint defect but does not eliminate every miss or partial box under all conditions.
- This change does not establish cross-camera identity, real OCR accuracy, physical hardware connectivity or external API behavior. Existing safety checks and earlier source changes are preserved.
- No camera setup, UI workflow, database data, original recordings, checksum logic or cloud delivery contract was changed.

## Verification commands

From `backend`, run `.venv/Scripts/python.exe scripts/validate_local_recordings.py --output ../docs/detector-fix-2026-10-13/candidate-heldout --detector-test D:/Cointainer.v1i.yolov8 --model models/acceptance-candidate.pt --confidence 0.35 --detector-only`.

Final application regressions used `.venv/Scripts/python.exe -m pytest -q` with temporary databases and mocked external services: **207 passed, 3 skipped, one existing Starlette deprecation warning in 52.62 seconds**. Final local settings/model load was verified to resolve the selected checkpoint, 640 image size, 0.35 threshold, empty aliases and the expected three model labels. `git diff --check` passed with existing Windows line-ending warnings only.
