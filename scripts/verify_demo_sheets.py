"""
Run every demo sheet through the live grading server (http://localhost:5001)
and print the actual results so you can confirm them before presenting.

Usage:
    # in another terminal:  python -m engine.server
    python scripts/verify_demo_sheets.py
"""

import json
import sys
from pathlib import Path
import requests

ROOT     = Path(__file__).resolve().parent.parent
SHEETS   = ROOT / "static" / "demo_sheets"
BASE_URL = "http://localhost:5001"

# Mirror the reference answers from generate_demo_sheets.py
REF = {
    "01_excellent_single.png": [
        "Photosynthesis is the process by which green plants convert light energy into chemical energy, using sunlight, water, and carbon dioxide to produce glucose and oxygen with the help of chlorophyll.",
    ],
    "02_partial_single.png": [
        "Photosynthesis is the process by which green plants convert light energy into chemical energy, using sunlight, water, and carbon dioxide to produce glucose and oxygen with the help of chlorophyll.",
    ],
    "03_multi_three_questions.png": [
        "Red blood cells transport oxygen from the lungs to body tissues by binding it to hemoglobin and remove carbon dioxide.",
        "Osmosis is the diffusion of water molecules across a semi-permeable membrane from a region of higher water concentration to lower water concentration.",
        "Newton's first law states that an object at rest stays at rest and an object in motion stays in motion at constant velocity unless acted upon by an external force.",
    ],
    "04_paraphrase_ablation.png": [
        "The mitochondrion is the powerhouse of the cell, generating ATP through oxidative phosphorylation in eukaryotic cells.",
    ],
    "05_wrong_answer_control.png": [
        "Gravity is the natural force of attraction between two masses, which on Earth pulls objects toward the planet's center.",
    ],
}


def grade(sheet_path: Path, refs: list[str], mode: str = "hybrid") -> dict:
    files = {"sheet": (sheet_path.name, sheet_path.read_bytes(), "image/png")}
    data  = {"grading_mode": mode}
    for i, r in enumerate(refs, 1):
        data[f"teacher_answer_{i}"] = r
    r = requests.post(f"{BASE_URL}/grade", files=files, data=data, timeout=600)
    r.raise_for_status()
    return r.json()


def ablation(student: str, teacher: str) -> dict:
    r = requests.post(
        f"{BASE_URL}/ablation",
        json={"student_text": student, "teacher_text": teacher},
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["ablation"]


def main():
    try:
        h = requests.get(f"{BASE_URL}/health", timeout=5).json()
        print(f"Server: {h['status']} · device={h['device']}\n")
    except Exception as e:
        print(f"Server unreachable at {BASE_URL} — start it with `python -m engine.server` first.")
        sys.exit(1)

    rows = []
    for name, refs in REF.items():
        path = SHEETS / name
        if not path.exists():
            print(f"  ⚠ missing: {path}")
            continue
        print(f"── Grading {name} …")
        result = grade(path, refs)

        if result.get("mode") == "multi":
            for q in result["per_question"]:
                rows.append([name, f"Q{q['question']}",
                             f"{q['similarity']*100:.1f}%",
                             f"{q['marks']}/10", q["grade"]])
            rows.append([name, "TOTAL",
                         f"—",
                         f"{result['combined_marks']}/{result['combined_total']}",
                         result["combined_grade"]])
        else:
            rows.append([name, "Q1",
                         f"{result['similarity']*100:.1f}%",
                         f"{result['marks']}/10", result["grade"]])

        # Print extracted OCR for transparency
        for i, b in enumerate(result.get("blocks", []), 1):
            print(f"   OCR[{i}]: {b!r}")

    # Pretty table
    print("\n" + "=" * 90)
    print(f"{'Sheet':<40}{'Q':<8}{'Sim':<10}{'Marks':<12}{'Grade':<14}")
    print("-" * 90)
    for r in rows:
        print(f"{r[0]:<40}{r[1]:<8}{r[2]:<10}{r[3]:<12}{r[4]:<14}")
    print("=" * 90)

    # Run the ablation specifically for the paraphrase sheet
    print("\n── Ablation comparison for 04_paraphrase_ablation.png ──")
    p = grade(SHEETS / "04_paraphrase_ablation.png", REF["04_paraphrase_ablation.png"])
    student_text = "\n".join(p["blocks"])
    print(f"Student (OCR): {student_text!r}")
    print(f"Teacher:       {REF['04_paraphrase_ablation.png'][0]!r}\n")
    abl = ablation(student_text, REF["04_paraphrase_ablation.png"][0])
    print(f"{'Mode':<10}{'Sim':<10}{'Marks':<12}{'Grade':<14}")
    print("-" * 46)
    for m in ("ml", "dl", "hybrid"):
        d = abl[m]
        print(f"{m.upper():<10}{d['similarity']*100:>5.1f}%   {d['marks']:>4}/10    {d['grade']:<14}")


if __name__ == "__main__":
    main()
