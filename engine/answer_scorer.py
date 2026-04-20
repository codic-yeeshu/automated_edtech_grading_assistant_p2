"""
Answer Scoring Engine — Phase 2

Three grading modes for ablation study:

  GradingMode.ML      (Model A — Advanced ML baseline)
    TF-IDF bigram cosine similarity (0.6) + Jaccard keyword overlap (0.4)
    No neural networks; interpretable and fast.

  GradingMode.DL      (Model B — Deep Learning)
    RoBERTa cross-encoder (0.7) + SBERT bi-encoder (0.3)
    Cross-encoder jointly encodes the (student, teacher) pair for
    full cross-attention — more accurate but slower than bi-encoder alone.

  GradingMode.HYBRID  (Model C — Proposed System)
    Cross-encoder (0.5) + BiLSTM quality scorer (0.3) + TF-IDF (0.2)
    DL captures deep semantics; TF-IDF ensures factual keyword coverage;
    BiLSTM adds structural answer quality signal.
"""

import re
import enum
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from engine.model_registry import ModelRegistry
from engine.lstm_quality import NeuralQualityScorer

logger = logging.getLogger(__name__)

# ─── Grading mode enum ────────────────────────────────────────────────────────

class GradingMode(str, enum.Enum):
    ML     = "ml"
    DL     = "dl"
    HYBRID = "hybrid"


# ─── Stop words ───────────────────────────────────────────────────────────────

_STOPWORDS = {
    "and", "the", "of", "in", "to", "a", "is", "are", "was", "were",
    "by", "on", "at", "for", "with", "this", "that", "it", "be", "an",
    "also", "but", "or", "not", "from", "as", "up", "set", "its", "has",
    "have", "had", "will", "can", "do", "did", "does", "so", "if", "then",
}

# ─── Lazy singleton for BiLSTM scorer ────────────────────────────────────────

_quality_scorer: NeuralQualityScorer | None = None


def _get_quality_scorer() -> NeuralQualityScorer:
    global _quality_scorer
    if _quality_scorer is None:
        _quality_scorer = NeuralQualityScorer()
    return _quality_scorer


# ─── Text normalisation ───────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ═══ ML Mode — TF-IDF + Jaccard ══════════════════════════════════════════════

def _tfidf_cosine(student: str, teacher: str) -> float:
    """
    TF-IDF bigram cosine similarity (Advanced ML baseline).

    TF-IDF weighs terms by their frequency in the document (TF) inversely
    proportional to how common they are across documents (IDF), so rare
    subject-specific terms receive higher weight.

    ngram_range=(1, 2) captures both unigrams and meaningful bigrams
    (e.g., 'photosynthesis process' vs separate 'photosynthesis' + 'process').
    """
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        tfidf_mat = vectorizer.fit_transform([student, teacher])
        return float(sk_cosine(tfidf_mat[0:1], tfidf_mat[1:2])[0][0])
    except Exception:
        return 0.0


def _jaccard_keyword(student: str, teacher: str) -> float:
    """
    Jaccard similarity on non-stopword unigrams.

    Ensures required factual terms (e.g. names, numbers, formulas) appear
    in the student answer even when meaning is paraphrased.
    """
    sw = set(_normalise(student).split()) - _STOPWORDS
    tw = set(_normalise(teacher).split()) - _STOPWORDS
    if not tw:
        return 1.0 if not sw else 0.0
    inter = sw & tw
    union = sw | tw
    return len(inter) / len(union) if union else 0.0


def ml_similarity(student: str, teacher: str) -> float:
    """ML similarity: TF-IDF cosine (0.6) + Jaccard keyword (0.4)."""
    return round(0.6 * _tfidf_cosine(student, teacher)
               + 0.4 * _jaccard_keyword(student, teacher), 4)


# ═══ DL Mode — Cross-encoder + SBERT ═════════════════════════════════════════

def _cross_encoder_score(student: str, teacher: str) -> float:
    """
    RoBERTa cross-encoder pairwise similarity.

    Unlike bi-encoders (which embed each text independently), the
    cross-encoder concatenates both texts as a single input:
        [CLS] student [SEP] teacher [SEP]
    allowing the model to use full cross-attention between the two texts.
    This is more accurate but cannot pre-compute embeddings.

    Model: cross-encoder/stsb-roberta-base (STS-B fine-tuned on sentence pairs)
    Output: continuous score ∈ [0, 1]
    """
    ce    = ModelRegistry.cross_encoder()
    score = ce.predict([(student, teacher)])[0]
    return float(max(0.0, min(1.0, float(score))))


def _sbert_cosine(student: str, teacher: str) -> float:
    """
    SBERT all-mpnet-base-v2 bi-encoder cosine similarity.

    Stronger baseline than all-MiniLM-L6-v2 used in Phase 1.
    """
    from sentence_transformers import util
    model      = ModelRegistry.sbert()
    embeddings = model.encode([student, teacher], convert_to_tensor=True)
    score      = util.cos_sim(embeddings[0], embeddings[1]).item()
    return float(max(0.0, min(1.0, score)))


def dl_similarity(student: str, teacher: str) -> float:
    """DL similarity: cross-encoder (0.7) + SBERT bi-encoder (0.3)."""
    return round(0.7 * _cross_encoder_score(student, teacher)
               + 0.3 * _sbert_cosine(student, teacher), 4)


# ═══ Hybrid Mode — Cross-encoder + BiLSTM + TF-IDF ═══════════════════════════

def hybrid_similarity(student: str, teacher: str) -> float:
    """
    Hybrid similarity (proposed system):

      Cross-encoder (0.50) — deep semantic understanding, full cross-attention
      BiLSTM quality  (0.30) — structural completeness via sentence-level LSTM
      TF-IDF cosine   (0.20) — factual term coverage guarantee

    The three components are complementary:
      - Cross-encoder covers meaning; TF-IDF covers vocabulary; BiLSTM covers structure.
    """
    ce      = _cross_encoder_score(student, teacher)
    quality = _get_quality_scorer().score(student, teacher)
    tfidf   = _tfidf_cosine(student, teacher)
    return round(0.50 * ce + 0.30 * quality + 0.20 * tfidf, 4)


# ─── Similarity → marks conversion ───────────────────────────────────────────

def similarity_to_marks(sim: float, total: int = 10) -> float:
    """
    Non-linear (curved) mapping from similarity score to marks out of `total`.

    Uses a piecewise linear scale that rewards high similarity
    disproportionately (a student who scores 0.85 gets ≥9/10).
    """
    if   sim >= 0.85: marks = 9.0 + (sim - 0.85) / 0.15
    elif sim >= 0.70: marks = 7.0 + (sim - 0.70) / 0.15 * 2
    elif sim >= 0.55: marks = 5.0 + (sim - 0.55) / 0.15 * 2
    elif sim >= 0.40: marks = 3.0 + (sim - 0.40) / 0.15 * 2
    else:             marks = max(0.0, sim / 0.40 * 3)
    return round(min(marks, float(total)), 1)


def assign_grade(marks: float, total: int = 10) -> str:
    """5-tier grading scale."""
    r = marks / total
    if r >= 0.85: return "Excellent"
    if r >= 0.70: return "Good"
    if r >= 0.55: return "Average"
    if r >= 0.40: return "Below Average"
    return "Poor"


def keyword_analysis(student: str, teacher: str):
    """Return (matched_keywords, missed_keywords) for feedback display."""
    sw      = set(_normalise(student).split()) - _STOPWORDS
    tw      = set(_normalise(teacher).split()) - _STOPWORDS
    matched = sorted(sw & tw)
    missed  = sorted(tw - sw)
    return matched, missed


# ─── Public API ───────────────────────────────────────────────────────────────

def evaluate_answer(
    student_text: str,
    teacher_text: str,
    mode: GradingMode = GradingMode.HYBRID,
    total: int = 10,
) -> dict:
    """
    Score a single student answer.

    Args:
        student_text: Transcribed student response
        teacher_text: Reference / model answer
        mode:         GradingMode.ML | .DL | .HYBRID
        total:        Maximum marks for this question

    Returns:
        dict with similarity, marks, grade, matched/missed keywords, mode
    """
    if not student_text.strip() or not teacher_text.strip():
        return {
            "similarity": 0.0, "marks": 0.0, "total": total,
            "grade": "Poor", "matched": [], "missed": [],
            "mode": mode.value,
        }

    if mode == GradingMode.ML:
        sim = ml_similarity(student_text, teacher_text)
    elif mode == GradingMode.DL:
        sim = dl_similarity(student_text, teacher_text)
    else:
        sim = hybrid_similarity(student_text, teacher_text)

    marks   = similarity_to_marks(sim, total)
    grade   = assign_grade(marks, total)
    matched, missed = keyword_analysis(student_text, teacher_text)

    return {
        "similarity": sim,
        "marks":      marks,
        "total":      total,
        "grade":      grade,
        "matched":    matched,
        "missed":     missed,
        "student":    student_text.strip(),
        "teacher":    teacher_text.strip(),
        "mode":       mode.value,
    }


def evaluate_multiple(
    question_map: dict,
    teacher_answers: list,
    mode: GradingMode = GradingMode.HYBRID,
    total_per_q: int = 10,
) -> dict:
    """
    Score all questions in a question_map and return combined result.

    Args:
        question_map:   {q_number: student_text}
        teacher_answers: list of reference answers (indexed 0 = Q1, etc.)
        mode:            GradingMode
        total_per_q:     Marks per question

    Returns:
        dict with per_question list + combined_marks, combined_total, combined_grade
    """
    per_question = []
    for i, q_num in enumerate(sorted(question_map.keys())):
        student_text = question_map[q_num]
        teacher_idx  = min(i, len(teacher_answers) - 1)
        teacher_text = teacher_answers[teacher_idx] if teacher_answers else ""

        result           = evaluate_answer(student_text, teacher_text, mode, total_per_q)
        result["question"] = q_num
        per_question.append(result)

    combined_marks = sum(r["marks"] for r in per_question)
    combined_total = total_per_q * len(per_question)
    combined_grade = (
        assign_grade(combined_marks, combined_total)
        if combined_total > 0 else "N/A"
    )

    return {
        "per_question":   per_question,
        "combined_marks": round(combined_marks, 1),
        "combined_total": combined_total,
        "combined_grade": combined_grade,
        "mode":           mode.value,
    }


def run_ablation(student_text: str, teacher_text: str, total: int = 10) -> dict:
    """
    Run all three modes on the same input pair. Used by the /ablation endpoint.
    Returns side-by-side comparison dict.
    """
    results = {}
    for m in GradingMode:
        r = evaluate_answer(student_text, teacher_text, m, total)
        results[m.value] = {
            "similarity": r["similarity"],
            "marks":      r["marks"],
            "grade":      r["grade"],
        }
    return results
