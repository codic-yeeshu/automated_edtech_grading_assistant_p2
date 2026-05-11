"""
Generate curated demo answer-sheets for the evaluator presentation.

Usage:
    python scripts/generate_demo_sheets.py
    # → writes 5 PNGs to static/demo_sheets/
    # → prints the demo plan (reference answers + expected scores) to stdout

Each sheet is rendered with a slightly informal hand-style font so TrOCR
(microsoft/trocr-base-handwritten) gives realistic transcripts while still
remaining legible enough that the demo never fails on bad OCR.
"""

import os
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT       = Path(__file__).resolve().parent.parent
OUT_DIR    = ROOT / "static" / "demo_sheets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Font discovery ───────────────────────────────────────────────────────────
# Chosen for legibility under TrOCR (handwritten checkpoint). Bradley Hand /
# Chalkduster look prettier but TrOCR hallucinates on them. Comic Sans gives
# a casual, slightly informal look while staying near-perfectly transcribed.
HAND_FONTS = [
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "/Library/Fonts/Comic Sans MS.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
PRINT_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


HAND_PATH  = _first_existing(HAND_FONTS)  or _first_existing(PRINT_FONTS)
PRINT_PATH = _first_existing(PRINT_FONTS) or HAND_PATH


def _font(path, size):
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ── Sheet renderer ───────────────────────────────────────────────────────────

W = 1700               # canvas width — wide so long answers fit in 2-3 lines max
MARGIN_X = 80
LINE_GAP = 32          # generous vertical gap so each line becomes its own
                        # OCR block (sheet_analyzer's row-smoothing window is
                        # 20 px — lines closer than that get merged and TrOCR
                        # is asked to read multi-line crops, which it fails on).
TITLE_SIZE = 40        # the "Q1." label only — not the question text
BODY_SIZE = 36
QUESTION_GAP = 60      # vertical gap between Q1, Q2, Q3 blocks
TITLE_COLOR = (30, 30, 30)
BODY_COLOR  = (40, 40, 40)


def _wrap(draw, text, font, max_width):
    """Word-wrap `text` to fit in `max_width` pixels for `font`."""
    words = text.split()
    lines, current = [], ""
    for w in words:
        candidate = (current + " " + w).strip()
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def render_sheet(filename: str, blocks: list[tuple[str, str]]):
    """
    Render a sheet with one or more (question_label, answer_text) blocks.

    `question_label` is rendered in the printed/title font (e.g. "Q1.").
    `answer_text` is rendered in the hand-style font, possibly across
    multiple wrapped lines.
    """
    title_font = _font(PRINT_PATH, TITLE_SIZE)
    body_font  = _font(HAND_PATH,  BODY_SIZE)

    # First pass: measure required height
    tmp_img = Image.new("RGB", (W, 100), "white")
    tmp_drw = ImageDraw.Draw(tmp_img)

    measured = []
    total_h  = 80  # top padding
    for label, ans in blocks:
        label_h = TITLE_SIZE + LINE_GAP
        wrapped = _wrap(tmp_drw, ans, body_font, W - 2 * MARGIN_X - 20)
        body_h  = len(wrapped) * (BODY_SIZE + LINE_GAP)
        measured.append((label, wrapped))
        total_h += label_h + body_h + QUESTION_GAP
    total_h += 80  # bottom padding

    # Final image
    img = Image.new("RGB", (W, total_h), "white")
    drw = ImageDraw.Draw(img)

    y = 80
    for label, wrapped in measured:
        drw.text((MARGIN_X, y), label, fill=TITLE_COLOR, font=title_font)
        y += TITLE_SIZE + LINE_GAP
        for line in wrapped:
            drw.text((MARGIN_X + 30, y), line, fill=BODY_COLOR, font=body_font)
            y += BODY_SIZE + LINE_GAP
        y += QUESTION_GAP

    out_path = OUT_DIR / filename
    img.save(out_path)
    return out_path


# ── Demo content ─────────────────────────────────────────────────────────────

REF_PHOTOSYNTHESIS = (
    "Photosynthesis is the process by which green plants convert light energy "
    "into chemical energy, using sunlight, water, and carbon dioxide to produce "
    "glucose and oxygen with the help of chlorophyll."
)

REF_MITOCHONDRIA = (
    "The mitochondrion is the powerhouse of the cell, generating ATP through "
    "oxidative phosphorylation in eukaryotic cells."
)

REF_GRAVITY = (
    "Gravity is the natural force of attraction between two masses, which on "
    "Earth pulls objects toward the planet's center."
)

REF_RBC = (
    "Red blood cells transport oxygen from the lungs to body tissues by binding "
    "it to hemoglobin and remove carbon dioxide."
)
REF_OSMOSIS = (
    "Osmosis is the diffusion of water molecules across a semi-permeable membrane "
    "from a region of higher water concentration to lower water concentration."
)
REF_NEWTON1 = (
    "Newton's first law states that an object at rest stays at rest and an object "
    "in motion stays in motion at constant velocity unless acted upon by an "
    "external force."
)


# Sheets contain the *student's answer only* — labelled "Q1.", "Q2.", "Q3.".
# The question prompt itself lives in the teacher's reference-answer textarea,
# not on the sheet. (Real exam sheets work this way: the question paper is
# separate from the answer booklet.) This also avoids the question text being
# OCR'd as a separate block and dragging the similarity score down.

SHEETS = [
    {
        "filename": "01_excellent_single.png",
        "blocks": [
            ("Q1.",
             "Photosynthesis is the process by which green plants use sunlight, "
             "water and carbon dioxide to produce glucose and release oxygen "
             "through chlorophyll in their leaves."),
        ],
        "mode": "hybrid",
        "question_prompts": ["Define photosynthesis."],
        "reference_answers": [REF_PHOTOSYNTHESIS],
        "expected": "Hybrid ≈ 0.85+ → 9/10 Excellent. Demo: high-quality answer + matched keywords list near-complete.",
    },
    {
        "filename": "02_partial_single.png",
        "blocks": [
            ("Q1.",
             "Photosynthesis is when plants make food using sunlight."),
        ],
        "mode": "hybrid",
        "question_prompts": ["Define photosynthesis."],
        "reference_answers": [REF_PHOTOSYNTHESIS],
        "expected": "Hybrid ≈ 0.45-0.65 → 4-6/10 Below Average / Average. Demo: partial credit + visible 'missed keywords' list.",
    },
    {
        "filename": "03_multi_three_questions.png",
        "blocks": [
            ("Q1.",
             "Red blood cells carry oxygen from the lungs to the body tissues "
             "using hemoglobin."),
            ("Q2.",
             "Osmosis is the movement of water from low concentration to high "
             "concentration through a semi-permeable membrane."),
            ("Q3.",
             "An object will keep moving unless a force stops it."),
        ],
        "mode": "hybrid",
        "question_prompts": [
            "What is the function of red blood cells?",
            "Define osmosis.",
            "State Newton's first law.",
        ],
        "reference_answers": [REF_RBC, REF_OSMOSIS, REF_NEWTON1],
        "expected": "Q1 Good/Excellent · Q2 Good (mostly correct) · Q3 Below Average / Average (oversimplified). Demo: multi-question mapping via Q-label regex + per-question breakdown UI + combined score.",
    },
    {
        "filename": "04_paraphrase_ablation.png",
        "blocks": [
            ("Q1.",
             "Cells obtain their chemical energy from these double-membrane "
             "organelles, which generate adenosine triphosphate through the "
             "process of aerobic cellular respiration."),
        ],
        "mode": "hybrid  (then click 'Compare All Models' for the ablation)",
        "question_prompts": ["What is the role of mitochondria?"],
        "reference_answers": [REF_MITOCHONDRIA],
        "expected": (
            "★ Star demo for ablation ★ — answer is semantically correct but "
            "shares almost no surface vocabulary with the reference. "
            "ML mode: ~0.05-0.20 Poor (TF-IDF/Jaccard see no overlap). "
            "DL mode: ~0.55-0.80 Average/Good (cross-encoder catches the meaning). "
            "Hybrid: in between. Use this slide to justify why DL beats keyword matching."
        ),
    },
    {
        "filename": "05_wrong_answer_control.png",
        "blocks": [
            ("Q1.",
             "Gravity is when birds fly south for the winter."),
        ],
        "mode": "hybrid",
        "question_prompts": ["Define gravity."],
        "reference_answers": [REF_GRAVITY],
        "expected": "Hybrid < 0.30 → 0-3/10 Poor across all modes. Demo: control case — system correctly refuses to over-credit a clearly wrong answer.",
    },
]


# ── Driver ───────────────────────────────────────────────────────────────────

def main():
    print(f"Hand font:  {HAND_PATH}")
    print(f"Print font: {PRINT_PATH}\n")

    print("Generating demo sheets …\n")
    for s in SHEETS:
        path = render_sheet(s["filename"], s["blocks"])
        print(f"  ✓ {path.relative_to(ROOT)}")

    # Demo plan ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("EVALUATOR DEMO PLAN")
    print("=" * 78)
    for i, s in enumerate(SHEETS, 1):
        print(f"\n── Demo {i}: {s['filename']} " + "─" * (60 - len(s['filename'])))
        print(f"Mode:               {s['mode']}")
        print(f"Question(s) you can read aloud (NOT on the sheet — sheet has answers only):")
        for j, q in enumerate(s.get("question_prompts", []), 1):
            print(f"  Q{j}: {q}")
        print(f"Reference answers (paste into the textareas, one per question):")
        for j, ref in enumerate(s["reference_answers"], 1):
            wrapped = textwrap.fill(ref, width=72,
                                    initial_indent=f"  Q{j}: ",
                                    subsequent_indent="       ")
            print(wrapped)
        print(f"Expected outcome:")
        print(textwrap.fill(s["expected"], width=72,
                            initial_indent="  ", subsequent_indent="  "))

    print("\n" + "=" * 78)
    print("Suggested presentation order: 1 → 2 → 5 → 3 → 4")
    print("(start strong, show partial credit, prove the system says 'Poor' when")
    print(" appropriate, demo multi-question, then close on the ablation win.)")
    print("=" * 78)


if __name__ == "__main__":
    main()
