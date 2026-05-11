"""
DL Grading Assistant — Flask Server (Phase 2)

Routes:
  GET  /              → evaluator dashboard
  POST /grade         → main evaluation endpoint (supports ML / DL / Hybrid mode)
  POST /ablation      → side-by-side comparison of all three grading modes
  GET  /health        → model status check
"""

import os
import sys
import base64
import logging
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
from dotenv import load_dotenv

# Ensure the project root (parent of engine/) is on sys.path so that
# `from engine.X import …` works regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from engine.sheet_analyzer import analyze_sheet, analyze_pil_sheet
from engine.answer_scorer  import evaluate_answer, evaluate_multiple, run_ablation, GradingMode

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Flask app ────────────────────────────────────────────────────────────────

_root        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_upload_dir  = os.path.join(_root, "static", "uploads")
os.makedirs(_upload_dir, exist_ok=True)

app = Flask(
    __name__,
    template_folder=os.path.join(_root, "templates"),
    static_folder=os.path.join(_root, "static"),
)
app.config["UPLOAD_FOLDER"]       = _upload_dir
app.config["MAX_CONTENT_LENGTH"]  = 16 * 1024 * 1024   # 16 MB

_ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


def _allowed(filename: str) -> bool:
    return _ext(filename) in _ALLOWED_EXTENSIONS


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("evaluator.html")


@app.route("/grade", methods=["POST"])
def grade():
    """
    Main grading endpoint.

    Form fields:
      sheet           : image file (PNG / JPG) or PDF
      teacher_answer_1, teacher_answer_2, …  : reference answers per question
      grading_mode    : "ml" | "dl" | "hybrid" (default: hybrid)
    """
    if "sheet" not in request.files:
        return jsonify({"error": "No answer sheet uploaded"}), 400

    sheet_file = request.files["sheet"]
    if not sheet_file.filename or not _allowed(sheet_file.filename):
        return jsonify({"error": "Upload a valid PNG, JPG, or PDF file"}), 400

    # ── Collect teacher answers ──────────────────────────────────────────────
    teacher_answers = []
    i = 1
    while True:
        val = request.form.get(f"teacher_answer_{i}", "").strip()
        if not val and i > 1:
            break
        if val:
            teacher_answers.append(val)
        elif i == 1:
            return jsonify({"error": "Provide at least one reference answer"}), 400
        i += 1

    # ── Determine grading mode ───────────────────────────────────────────────
    mode_str = request.form.get("grading_mode", "hybrid").lower()
    try:
        mode = GradingMode(mode_str)
    except ValueError:
        mode = GradingMode.HYBRID

    # ── Save uploaded file ───────────────────────────────────────────────────
    # Prefix with a millisecond timestamp so repeat uploads of the same name
    # never overwrite a file we (or another in-flight request) are still reading.
    import time
    filename  = f"{int(time.time() * 1000)}_{secure_filename(sheet_file.filename)}"
    save_path = os.path.join(_upload_dir, filename)
    sheet_file.save(save_path)

    # ── OCR pipeline ─────────────────────────────────────────────────────────
    try:
        if _ext(filename) == "pdf":
            pages          = convert_from_path(save_path, dpi=200)
            all_texts      = []
            all_questions  = {}
            annotated_imgs = []

            for page in pages:
                ann_bytes, blk_texts, q_map = analyze_pil_sheet(page)
                all_texts.extend(blk_texts)
                offset = max(all_questions.keys(), default=0)
                for q, text in q_map.items():
                    key = q + offset if all_questions else q
                    all_questions[key] = (
                        all_questions[key] + "\n" + text if key in all_questions else text
                    )
                annotated_imgs.append(base64.b64encode(ann_bytes).decode("utf-8"))

            block_texts  = all_texts
            question_map = all_questions
            img_list     = annotated_imgs
        else:
            ann_bytes, block_texts, question_map = analyze_sheet(save_path)
            img_list = [base64.b64encode(ann_bytes).decode("utf-8")]

    except Exception as exc:
        logger.exception("OCR pipeline failed")
        return jsonify({"error": f"Analysis failed: {exc}"}), 500

    # ── Scoring ──────────────────────────────────────────────────────────────
    if len(teacher_answers) > 1:
        result = evaluate_multiple(question_map, teacher_answers, mode)
        return jsonify({
            "mode":           "multi",
            "grading_mode":   mode.value,
            "images":         img_list,
            "blocks":         block_texts,
            "per_question":   result["per_question"],
            "combined_marks": result["combined_marks"],
            "combined_total": result["combined_total"],
            "combined_grade": result["combined_grade"],
        })
    else:
        student_text = "\n".join(block_texts)
        result       = evaluate_answer(student_text, teacher_answers[0], mode)
        return jsonify({
            "mode":         "single",
            "grading_mode": mode.value,
            "images":       img_list,
            "blocks":       block_texts,
            "similarity":   result["similarity"],
            "marks":        result["marks"],
            "total":        result["total"],
            "grade":        result["grade"],
            "matched":      result["matched"],
            "missed":       result["missed"],
        })


@app.route("/ablation", methods=["POST"])
def ablation():
    """
    Ablation study endpoint.
    Runs all three grading modes on the same (student, teacher) text pair
    and returns a side-by-side comparison.

    JSON body:
      { "student_text": "...", "teacher_text": "..." }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    student = data.get("student_text", "").strip()
    teacher = data.get("teacher_text", "").strip()
    if not student or not teacher:
        return jsonify({"error": "student_text and teacher_text are required"}), 400

    try:
        comparison = run_ablation(student, teacher)
    except Exception as exc:
        logger.exception("Ablation failed")
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "ablation": comparison,
        "student":  student,
        "teacher":  teacher,
    })


@app.route("/health")
def health():
    """Returns model load status for monitoring."""
    from engine.model_registry import ModelRegistry
    return jsonify({
        "status":         "ok",
        "trocr_loaded":   ModelRegistry._trocr_model is not None,
        "sbert_loaded":   ModelRegistry._sbert_model is not None,
        "ce_loaded":      ModelRegistry._cross_encoder is not None,
        "device":         str(ModelRegistry.device()),
    })


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5001)
