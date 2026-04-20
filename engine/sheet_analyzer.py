"""
Document Analysis Pipeline — Phase 2

Key upgrade over Phase 1:
  OCR engine switched from EasyOCR → TrOCR (microsoft/trocr-base-handwritten)
  TrOCR is a Vision Encoder-Decoder model:
    Encoder: ViT processes the image as a sequence of 16×16 patches
    Decoder: GPT-2-style transformer generates characters autoregressively
    Pre-trained on IAM handwriting, SROIE, and other benchmark datasets

Block detection still uses the reliable OpenCV horizontal projection
histogram approach (no labeled training data required, fast on CPU).
"""

import re
import cv2
import numpy as np
import logging
import torch
from PIL import Image
from scipy.ndimage import uniform_filter1d

from engine.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# ─── Tuning constants ─────────────────────────────────────────────────────────
MIN_BLOCK_HEIGHT    = 40       # pixels — skip blocks shorter than this
MIN_WIDTH_FRACTION  = 0.30     # fraction of image width — skip narrow artefacts
ROW_SMOOTHING_SIZE  = 20       # uniform-filter window for row projection
INK_THRESHOLD_RATIO = 0.02     # fraction of peak projection treated as "ink"
CROP_PADDING        = 8        # pixels added around each block before OCR

# ─── Question label pattern (same robust regex as Phase 1) ───────────────────
Q_LABEL_RE = re.compile(
    r"^\s*"
    r"(?:"
    r"(?:[Qq](?:uestion)?|[Aa](?:ns(?:wer)?)?)[.\s]*(\d+)"   # Q1, Ans 1
    r"|"
    r"\((\d+)\)"                                               # (1)
    r"|"
    r"(\d+)\s*[).:]\s*"                                        # 1) 1. 1:
    r")"
)

# ─── Annotation colour palette ────────────────────────────────────────────────
_PALETTE = [
    (220, 60, 60),  (60, 160, 220), (60, 200, 80),
    (220, 150, 40), (160, 60, 220), (40, 200, 200),
]


# ─── Image preprocessing ──────────────────────────────────────────────────────

def _binarize(img_bgr: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian threshold: ink → white, background → black."""
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray   = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25, C=10,
    )
    return binary


# ─── Block detection (horizontal projection histogram) ───────────────────────

def locate_answer_blocks(img_bgr: np.ndarray) -> list:
    """
    Find answer regions via horizontal ink projection.

    Algorithm:
      1. Binarize image (ink white, background black)
      2. Sum ink pixels across each row → 1-D projection vector
      3. Smooth with uniform filter to bridge intra-word gaps
      4. Threshold at 2% of peak → extract contiguous "band" regions
      5. Filter bands by minimum height and minimum width
      6. Return list of (x1, y1, x2, y2) bounding boxes
    """
    binary = _binarize(img_bgr)
    h, w   = binary.shape

    row_ink = uniform_filter1d(
        np.sum(binary, axis=1).astype(float),
        size=ROW_SMOOTHING_SIZE,
    )
    threshold = row_ink.max() * INK_THRESHOLD_RATIO

    in_block, start, bands = False, 0, []
    for i, val in enumerate(row_ink):
        if val > threshold and not in_block:
            start, in_block = i, True
        elif val <= threshold and in_block:
            bands.append((start, i))
            in_block = False
    if in_block:
        bands.append((start, h))

    min_w  = int(w * MIN_WIDTH_FRACTION)
    blocks = []
    for y1, y2 in bands:
        if y2 - y1 < MIN_BLOCK_HEIGHT:
            continue
        col_sum = np.sum(binary[y1:y2, :], axis=0)
        nz = np.where(col_sum > 0)[0]
        if len(nz) == 0:
            continue
        x1 = max(0, int(nz[0])  - 10)
        x2 = min(w, int(nz[-1]) + 10)
        if x2 - x1 < min_w:
            continue
        blocks.append((x1, y1, x2, y2))
    return blocks


# ─── TrOCR inference ─────────────────────────────────────────────────────────

def _upscale_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Scale small crops up so TrOCR patch tokenizer gets enough detail."""
    ch, cw = crop_bgr.shape[:2]
    scale = 2.5 if (ch < 100 or cw < 300) else (1.8 if ch < 200 else 1.0)
    if scale > 1.0:
        crop_bgr = cv2.resize(
            crop_bgr, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    return crop_bgr


def trocr_read_block(img_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> str:
    """
    Run TrOCR on a single bounding-box crop.

    TrOCR inference:
      1. Crop + upscale
      2. Convert BGR → RGB → PIL.Image
      3. TrOCRProcessor tokenizes image into 16×16 ViT patches
      4. VisionEncoderDecoderModel.generate() uses beam search (4 beams)
         to decode the patch sequence into text tokens autoregressively
      5. Decode token IDs → UTF-8 string

    Beam search (num_beams=4) gives better accuracy than greedy decoding
    at ~4× inference cost — acceptable for single block CPU inference.
    """
    h, w      = img_bgr.shape[:2]
    crop_bgr  = img_bgr[
        max(0, y1 - CROP_PADDING): min(h, y2 + CROP_PADDING),
        max(0, x1 - CROP_PADDING): min(w, x2 + CROP_PADDING),
    ]
    crop_bgr  = _upscale_crop(crop_bgr)
    pil_image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

    processor, model = ModelRegistry.trocr()
    pixel_values     = processor(images=pil_image, return_tensors="pt").pixel_values
    pixel_values     = pixel_values.to(ModelRegistry.device())

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


# ─── Question label parsing ───────────────────────────────────────────────────

def extract_question_label(text: str):
    """
    Parse question number from the first line of block text.
    Returns (q_number, cleaned_text) or (None, original_text).
    """
    if not text or not text.strip():
        return None, text

    lines      = text.strip().split("\n")
    match      = Q_LABEL_RE.match(lines[0])
    if not match:
        return None, text

    q_num          = int(match.group(1) or match.group(2) or match.group(3))
    cleaned_first  = Q_LABEL_RE.sub("", lines[0]).strip()
    if cleaned_first:
        lines[0] = cleaned_first
    else:
        lines = lines[1:]
    return q_num, "\n".join(lines).strip()


def _split_merged_block(text: str) -> list:
    """
    Handle blocks that contain multiple question answers (merged by OCR).
    Returns list of (q_num, text) tuples.
    """
    if not text or not text.strip():
        return []

    lines, segments            = text.strip().split("\n"), []
    current_q, current_lines   = None, []

    for line in lines:
        match = Q_LABEL_RE.match(line)
        if match:
            if current_lines:
                segments.append((current_q, "\n".join(current_lines).strip()))
            current_q     = int(match.group(1) or match.group(2) or match.group(3))
            cleaned       = Q_LABEL_RE.sub("", line).strip()
            current_lines = [cleaned] if cleaned else []
        else:
            current_lines.append(line)

    if current_lines:
        segments.append((current_q, "\n".join(current_lines).strip()))
    return segments


def map_blocks_to_questions(block_texts: list) -> dict:
    """
    Assign question numbers to OCR'd block texts.

    Handles:
      - Multiple blocks with same Q-label  → merge
      - Single block with multiple Q-labels → split
      - No labels at all                   → sequential (Block 1 = Q1)

    Returns: {q_number: combined_text}
    """
    questions   = {}
    has_label   = False

    for block_text in block_texts:
        segments = _split_merged_block(block_text)

        if len(segments) > 1 and any(s[0] is not None for s in segments):
            has_label = True
            for q_num, text in segments:
                if q_num is not None and text:
                    questions.setdefault(q_num, []).append(text)
        else:
            q_num, cleaned = extract_question_label(block_text)
            if q_num is not None:
                has_label = True
                questions.setdefault(q_num, []).append(cleaned)
            else:
                questions.setdefault(None, []).append(block_text)

    if not has_label:
        return {i + 1: t for i, t in enumerate(block_texts) if t.strip()}

    if None in questions:
        unlabeled = questions.pop(None)
        if questions:
            questions[max(questions.keys())].extend(unlabeled)
        else:
            for i, text in enumerate(unlabeled):
                questions[i + 1] = [text]

    return {q: "\n".join(texts) for q, texts in sorted(questions.items())}


# ─── Annotation ───────────────────────────────────────────────────────────────

def draw_annotations(img_bgr: np.ndarray, blocks: list) -> np.ndarray:
    """Overlay coloured bounding boxes with 'Answer N' labels."""
    out       = img_bgr.copy()
    h, w      = out.shape[:2]
    font      = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(2, w // 400)
    fscale    = max(0.55, w / 2000)

    for idx, (x1, y1, x2, y2) in enumerate(blocks):
        color          = _PALETTE[idx % len(_PALETTE)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label          = f"Answer {idx + 1}"
        (tw, th), base = cv2.getTextSize(label, font, fscale, thickness)
        ly             = max(y1 - 6, th + 4)
        cv2.rectangle(out, (x1, ly - th - 4), (x1 + tw + 6, ly + base), color, cv2.FILLED)
        cv2.putText(out, label, (x1 + 3, ly), font, fscale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


# ─── Public pipeline functions ────────────────────────────────────────────────

def analyze_sheet(image_path: str):
    """
    Full pipeline for a single image file.
    Returns (annotated_png_bytes, block_texts, question_map).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return _run_pipeline(img)


def analyze_pil_sheet(pil_img):
    """Full pipeline for a PIL Image (PDF page)."""
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return _run_pipeline(img)


def _run_pipeline(img_bgr: np.ndarray):
    """
    Core pipeline:
      locate_answer_blocks → trocr_read_block (per block) →
      map_blocks_to_questions → draw_annotations → encode PNG
    """
    blocks       = locate_answer_blocks(img_bgr)
    texts        = [trocr_read_block(img_bgr, *b) for b in blocks]
    question_map = map_blocks_to_questions(texts)
    annotated    = draw_annotations(img_bgr, blocks)
    _, buf       = cv2.imencode(".png", annotated)
    return buf.tobytes(), texts, question_map
