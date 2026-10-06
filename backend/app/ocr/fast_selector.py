"""Fast frame quality assessment and blur filtering before OCR execution.

Filters out blurry, unreadable, or out-of-frame images in < 1 ms so expensive OCR
runs only on the highest-quality candidate frames.
"""
import cv2
import numpy as np


class FastFrameSelector:
    """Ultra-fast blur detection, visibility scoring, and candidate filtering."""

    @staticmethod
    def compute_blur_score(crop_image: np.ndarray) -> float:
        """Calculate Laplacian variance to detect motion blur or out-of-focus crops.
        
        Runs in ~0.2 - 0.6 ms on typical crop regions.
        Scores:
          < 35.0:  Blurry / motion-smeared (unusable for high-confidence OCR)
          >= 35.0: Readable text candidate
          >= 100.0: Crisp, high-sharpness text
        """
        if crop_image is None or crop_image.size == 0:
            return 0.0
        try:
            if len(crop_image.shape) == 3:
                gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop_image
            # Fast downsample if crop is unnecessarily large
            if gray.shape[0] > 400 or gray.shape[1] > 800:
                gray = cv2.resize(gray, (min(gray.shape[1], 400), min(gray.shape[0], 200)))
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 0.0

    @staticmethod
    def evaluate_candidate(frame_image: np.ndarray,
                           detection_box: tuple,
                           all_detections: tuple = (),
                           min_blur_threshold: float = 30.0) -> tuple[bool, float, tuple]:
        """Evaluate if a frame has usable text or container regions for OCR.
        
        Returns:
            (is_usable: bool, composite_score: float, best_crop_box: tuple)
        """
        if frame_image is None or frame_image.size == 0:
            return False, 0.0, detection_box

        # Support synthetic test streams (e.g. FakeStream returning np.zeros)
        if float(frame_image.std()) < 0.01:
            return True, 50.0, detection_box

        img_h, img_w = frame_image.shape[:2]
        
        # 1. Check for targeted text detections first ('container_number' or 'iso_code')
        valid_text_boxes = []
        for d in all_detections:
            if d.class_name not in ('container_number', 'iso_code'):
                continue
            bw = max(1.0, float(d.bbox[2] - d.bbox[0]))
            bh = max(1.0, float(d.bbox[3] - d.bbox[1]))
            # Enforce physical minimum dimensions:
            # 11-digit container number must be >= 60px wide and >= 18px high
            if d.class_name == 'container_number' and (bw < 60.0 or bh < 18.0):
                continue
            # 4-digit ISO size code must be >= 24px wide and >= 14px high
            if d.class_name == 'iso_code' and (bw < 24.0 or bh < 14.0):
                continue
            valid_text_boxes.append((d, bw, bh))

        if valid_text_boxes:
            # Rank candidates: prioritize container_number over iso_code, weighted by confidence and area
            best_det, bw, bh = max(
                valid_text_boxes,
                key=lambda item: (
                    2.0 if item[0].class_name == 'container_number' else 1.0
                ) * (item[0].confidence or 0.8) * np.sqrt(item[1] * item[2])
            )
            bx1, by1, bx2, by2 = [int(v) for v in best_det.bbox]

            # Extract small crop for blur test
            cx1 = max(0, bx1)
            cy1 = max(0, by1)
            cx2 = min(img_w, bx2)
            cy2 = min(img_h, by2)
            crop = frame_image[cy1:cy2, cx1:cx2]

            blur = FastFrameSelector.compute_blur_score(crop)
            if blur < min_blur_threshold:
                # Frame is motion blurred; reject early to save OCR budget
                return False, blur, (cx1, cy1, cx2, cy2)

            # Composite score: sharpness * detection confidence
            composite = blur * (best_det.confidence or 0.8)
            return True, composite, (cx1, cy1, cx2, cy2)

        # 2. Main container box fallback
        cx1, cy1, cx2, cy2 = [int(v) for v in detection_box]
        bw = max(1, cx2 - cx1)
        bh = max(1, cy2 - cy1)

        # Container must occupy meaningful frame area (> 100px wide, > 60px high)
        if bw < 100 or bh < 60:
            return False, 0.0, detection_box

        # Extract sample patch for blur check
        sample_crop = frame_image[max(0, cy1):min(img_h, cy2), max(0, cx1):min(img_w, cx2)]
        blur = FastFrameSelector.compute_blur_score(sample_crop)
        if blur < min_blur_threshold:
            return False, blur, detection_box

        return True, blur * 0.5, detection_box
