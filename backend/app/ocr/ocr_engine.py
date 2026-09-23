from typing import Protocol, Any
import logging
from app.domain import OCRRead

log = logging.getLogger('gate')


class OCREngine(Protocol):
    def read(self, image: Any) -> OCRRead: ...


def _extract_text_from_items(items):
    if not items:
        return '', 0.0
    # Check vertical columns first
    cols = {}
    for it in items:
        xc = (it[0][0][0] + it[0][1][0]) / 2
        yc = (it[0][0][1] + it[0][2][1]) / 2
        matched_c = next((c for c in cols if abs(c - xc) < 55), None)
        if matched_c is None:
            matched_c = xc
            cols[matched_c] = []
        cols[matched_c].append(it)

    import re
    SIX_PREFIX = re.compile(r'(?<![A-Z0-9])([A-Z][\s-]*[A-Z][\s-]*[A-Z][\s-]*[UJZ](?:[\s-]*[0-9]){6})(?![A-Z0-9])')
    from app.ocr.validator import CANDIDATE, STRUCTURE, correct_feet_size_codes
    for c in sorted(cols.keys()):
        chars = sorted(cols[c], key=lambda it: it[0][0][1])
        col_text = ''.join(str(it[1]) for it in chars)
        if STRUCTURE.search(col_text) or CANDIDATE.search(col_text) or SIX_PREFIX.search(col_text):
            combined = ' '.join(str(it[1]) for it in chars)
            min_conf = float(min(it[2] for it in chars)) if chars else 0.0
            return correct_feet_size_codes(combined)[:4096], min_conf

    # Standard horizontal reading order
    h_items = sorted(items, key=lambda it: (it[0][0][1] // 30, it[0][0][0]))
    combined = ' '.join(str(it[1]) for it in h_items)
    min_conf = float(min(it[2] for it in h_items)) if h_items else 0.0
    return correct_feet_size_codes(combined)[:4096], min_conf


class EasyOCREngine:
    def __init__(self, settings):
        import easyocr
        self.reader = easyocr.Reader(['en'], gpu=settings.ocr_gpu,
                                     model_storage_directory=str(settings.ocr_model_directory),
                                     download_enabled=settings.ocr_download_enabled, verbose=False)

    def read(self, image) -> OCRRead:
        import re
        import cv2
        from app.ocr.validator import CANDIDATE, STRUCTURE
        SIX_PREFIX = re.compile(r'(?<![A-Z0-9])([A-Z][\s-]*[A-Z][\s-]*[A-Z][\s-]*[UJZ](?:[\s-]*[0-9]){6})(?![A-Z0-9])')
        DIGIT_PREFIX = re.compile(r'(?<![A-Z0-9])([A-Z0-9][\s-]*[A-Z0-9][\s-]*[A-Z0-9][\s-]*[UJZ0-9](?:[\s-]*[0-9]){6,7})(?![A-Z0-9])')
        if image is None or getattr(image, 'size', 0) == 0:
            return OCRRead('', 0.0)

        h, w = image.shape[:2]
        # Avoid processing massive full 4K/1080p crops without downscaling
        proc_img = image
        if max(h, w) > 1280:
            scale = 1280.0 / max(h, w)
            proc_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = proc_img.shape[:2]

        is_vertical = h > 1.5 * w

        if is_vertical:
            if w < 100:
                scale = min(3.0, 120.0 / max(1, w))
                proc_img = cv2.resize(proc_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            proc_img = cv2.copyMakeBorder(proc_img, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # 1. Fast standard 0-degree pass
        items = self.reader.readtext(
            proc_img, detail=1, paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            rotation_info=None,
            link_threshold=0.9 if is_vertical else 0.2
        )

        txt, conf = _extract_text_from_items(items)
        if STRUCTURE.search(txt) or CANDIDATE.search(txt) or SIX_PREFIX.search(txt) or DIGIT_PREFIX.search(txt):
            return OCRRead(txt, conf)

        # If not vertical and no valid code found yet, check vertical post strips
        if not is_vertical and w > 200:
            strips = [
                proc_img[:, int(w * 0.60):],
                proc_img[int(h * 0.08):int(h * 0.90), int(w * 0.35):int(w * 0.80)]
            ]
            for strip in strips:
                if strip.shape[0] < 50 or strip.shape[1] < 50:
                    continue
                items_strip = self.reader.readtext(
                    strip, detail=1, paragraph=False,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    link_threshold=0.9
                )
                txt_strip, conf_strip = _extract_text_from_items(items_strip)
                if STRUCTURE.search(txt_strip) or CANDIDATE.search(txt_strip) or SIX_PREFIX.search(txt_strip) or DIGIT_PREFIX.search(txt_strip):
                    return OCRRead(txt_strip, conf_strip)

        # For vertical crops: if padded read didn't find a candidate, retry without padding.
        if is_vertical:
            base_img = image
            if w < 100:
                scale = min(3.0, 120.0 / max(1, w))
                base_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            items2 = self.reader.readtext(
                base_img, detail=1, paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                link_threshold=0.9
            )
            txt2, conf2 = _extract_text_from_items(items2)
            if STRUCTURE.search(txt2) or CANDIDATE.search(txt2) or SIX_PREFIX.search(txt2) or DIGIT_PREFIX.search(txt2):
                return OCRRead(txt2, conf2)
            if len(txt2) > len(txt):
                txt, conf = txt2, conf2

        if not items and is_vertical:
            rot_cw = cv2.rotate(proc_img, cv2.ROTATE_90_CLOCKWISE)
            items = self.reader.readtext(
                rot_cw, detail=1, paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            txt, conf = _extract_text_from_items(items)

        return OCRRead(txt, conf)


class OlmOCREngine:
    """
    oLmOCR / Qwen2-VL Vision-Language OCR Engine.
    Specialized for container code extraction across complex orientations
    (vertical text, stacked characters, low-contrast stamps).
    """
    def __init__(self, settings):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.model_name = getattr(settings, 'ocr_model_name', 'Qwen/Qwen2-VL-2B-Instruct')
        self.device = 'cuda' if (getattr(settings, 'ocr_gpu', True) and torch.cuda.is_available()) else 'cpu'
        self.torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32

        log.info(f'Loading oLmOCR (Qwen2-VL) model {self.model_name} on {self.device}...')
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device
        )
        self.prompt = (
            "You are an industrial container OCR system. Read all container markings on this image: "
            "the 4-letter owner code (ending in U, J, or Z), 6-digit serial number, 1-digit check digit, and 4-character ISO size-type code (e.g. 45G1, 22G1). "
            "Output ONLY the letters and digits visible on the container separated by spaces. Do NOT invent or guess codes. If no container markings are clearly visible, output empty."
        )
        log.info(f'oLmOCR model loaded successfully on {self.device}.')

    def read(self, image) -> OCRRead:
        import cv2
        import re
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        if image is None or getattr(image, 'size', 0) == 0:
            return OCRRead('', 0.0)

        try:
            h, w = image.shape[:2]
            if max(h, w) > 768:
                scale = 768.0 / max(h, w)
                proc_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                proc_img = image

            rgb_image = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            messages = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': pil_image},
                        {'type': 'text', 'text': self.prompt}
                    ]
                }
            ]

            text_template = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_template],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0].strip()

            if not output_text:
                return OCRRead('', 0.0)

            # Strip conversational text if model produced extra narrative
            cleaned = output_text.replace('\n', ' ').replace('-', ' ')
            code_match = re.search(r'([A-Z]{3,4}(?:[\s-]*[0-9]){6,7}(?:[\s-]*[1-4LMN][0-9][A-Z0-9]{2})?)', cleaned.upper())
            if code_match:
                output_text = code_match.group(1).strip()
            else:
                output_text = cleaned.strip()

            from app.ocr.validator import correct_feet_size_codes
            output_text = correct_feet_size_codes(output_text)

            has_letters = any(c.isalpha() for c in output_text)
            has_digits = any(c.isdigit() for c in output_text)
            if has_letters and has_digits:
                confidence = 0.95
            elif has_digits:
                confidence = 0.85
            else:
                confidence = 0.70

            return OCRRead(output_text[:4096], confidence)

        except Exception as exc:
            log.error(f'oLmOCR read error: {exc}', exc_info=True)
            return OCRRead('', 0.0)


def create_ocr_engine(settings) -> OCREngine:
    engine_type = getattr(settings, 'ocr_engine', 'easyocr').lower()
    if engine_type in ('olmocr', 'qwen2_vl', 'qwen2vl', 'qwen'):
        return OlmOCREngine(settings)
    return EasyOCREngine(settings)
