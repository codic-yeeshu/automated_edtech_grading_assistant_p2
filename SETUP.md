# Setup & Walkthrough Guide

## Project Overview

This is **Phase 2** of the Automated EdTech Grading Assistant. It replaces the Phase 1
rule-based OCR pipeline with a deep-learning stack:

---

## Directory Structure

```
/
├── engine/
│   ├── server.py            # Flask web server — main entry point
│   ├── sheet_analyzer.py    # OpenCV block detection + TrOCR inference
│   ├── answer_scorer.py     # Multi-mode grading (ML / DL / Hybrid)
│   ├── model_registry.py    # Lazy-loaded DL model singleton cache
│   ├── lstm_quality.py      # BiLSTM + attention PyTorch module
│   └── requirements.txt     # Python dependencies
├── templates/
│   └── evaluator.html       # Responsive single-page UI
├── static/
│   └── uploads/             # Auto-created at runtime
├── notebooks/
│   ├── 01_trocr_ocr.ipynb        # TrOCR vs EasyOCR comparison
│   ├── 02_grading_models.ipynb   # Semantic model comparison
│   └── 03_ablation_study.ipynb   # Full ablation with metrics
├── SETUP.md     ← you are here
```

---

## Prerequisites

- Python 3.10 or higher
- pip 23+
- ~3 GB disk space (for model weights)
- macOS / Linux / Windows with WSL2
- **No GPU required** — all models run on CPU (GPU auto-detected if available)

For PDF support, install `poppler`:
```bash
# macOS
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils
```

---

## Installation

### Step 1 — Clone / navigate to project

```bash
cd DL_Grading_Assistant
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
# venv\Scripts\activate.bat    # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **First run note**: TrOCR (~400 MB), SBERT (~420 MB), and RoBERTa cross-encoder (~500 MB)
> are downloaded automatically on first use. Subsequent runs use the local HuggingFace cache.

### Step 4 — (Optional) Create `.env`

```bash
cp .env.example .env
```

The default `.env` is empty; no API keys are required.

---

## Running the Application

```bash
cd engine
python server.py
```

The server starts at **http://localhost:5001**.

Open the URL in any modern browser.

---

## Usage Walkthrough

### Basic Single-Question Grading

1. **Upload** — Click the upload zone or drag and drop a PNG/JPG answer sheet.
2. **Select mode** — Choose **Model C — Hybrid** (recommended) or ML/DL.
3. **Enter reference answer** — Type the correct answer in the "Question 1" field.
4. **Click "Grade Answer Sheet"**.
5. **View results**:
   - Score out of 10 with grade badge (Excellent / Good / Average / Below Average / Poor)
   - Matched and missed keywords
   - Annotated image showing detected answer blocks
   - Raw extracted text from TrOCR

### Multi-Question Grading

1. Click **+ Add Question** to add more reference answer fields.
2. Upload a sheet where each answer begins with `Q1`, `Q.1`, `1)`, `(1)`, etc.
3. The system automatically maps each detected block to the corresponding question.
4. Results show per-question scores plus a combined total.

### Ablation Study

1. Grade a sheet first (to populate the extracted text).
2. Click **"Compare All Models"**.
3. The ablation table shows Model A / B / C similarity scores and grades side-by-side.
4. The highlighted row indicates the best-performing model for that input.

### PDF Upload

- Upload a multi-page PDF exam.
- Each page is processed at 200 DPI.
- Question numbers are tracked across pages.

---

## API Endpoints

### `POST /grade`

**Form fields:**
| Field | Type | Description |
|-------|------|-------------|
| `sheet` | file | PNG, JPG, or PDF answer sheet |
| `teacher_answer_1` | string | Reference answer for Q1 |
| `teacher_answer_2` | string | Reference answer for Q2 (optional) |
| `grading_mode` | string | `ml` / `dl` / `hybrid` (default: `hybrid`) |

**Response (single mode):**
```json
{
  "mode": "single",
  "grading_mode": "hybrid",
  "images": ["<base64>"],
  "blocks": ["extracted text..."],
  "similarity": 0.812,
  "marks": 8.0,
  "total": 10,
  "grade": "Good",
  "matched": ["photosynthesis", "glucose"],
  "missed": ["chlorophyll"]
}
```

**Response (multi mode):**
```json
{
  "mode": "multi",
  "grading_mode": "hybrid",
  "per_question": [
    {"question": 1, "marks": 8.0, "total": 10, "grade": "Good", ...},
    {"question": 2, "marks": 6.5, "total": 10, "grade": "Average", ...}
  ],
  "combined_marks": 14.5,
  "combined_total": 20,
  "combined_grade": "Good"
}
```

### `POST /ablation`

**JSON body:**
```json
{
  "student_text": "Plants use sunlight to make food...",
  "teacher_text": "Photosynthesis is the process..."
}
```

**Response:**
```json
{
  "ablation": {
    "ml":     {"similarity": 0.612, "marks": 6.1, "grade": "Average"},
    "dl":     {"similarity": 0.741, "marks": 7.4, "grade": "Good"},
    "hybrid": {"similarity": 0.779, "marks": 7.8, "grade": "Good"}
  }
}
```

### `GET /health`

Returns model load status:
```json
{
  "status": "ok",
  "trocr_loaded": true,
  "sbert_loaded": true,
  "ce_loaded": true,
  "device": "cpu"
}
```

---

## Running the Notebooks

```bash
pip install jupyter
cd notebooks
jupyter notebook
```

Open notebooks in order:
1. `01_trocr_ocr.ipynb` — TrOCR architecture and beam search analysis
2. `02_grading_models.ipynb` — Compare all three grading models
3. `03_ablation_study.ipynb` — Full ablation with MAE and Pearson r metrics

---

## Model Download Details

| Model | HuggingFace ID | Size | Purpose |
|-------|---------------|------|---------|
| TrOCR | `microsoft/trocr-base-handwritten` | ~400 MB | Handwriting OCR |
| SBERT | `all-mpnet-base-v2` | ~420 MB | Bi-encoder semantic similarity |
| Cross-encoder | `cross-encoder/stsb-roberta-base` | ~500 MB | Pairwise semantic scoring |

All models are cached in `~/.cache/huggingface/hub/` after first download.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `pdf2image` error | Install `poppler` (see Prerequisites) |
| TrOCR slow on CPU | Expected — ~5–15s per block on CPU. Use GPU if available. |
| `transformers` not found | `pip install transformers accelerate` |
| Out of memory | Reduce image DPI from 200 to 150 in `server.py` |
| Port 5001 in use | Change port in `server.py`: `app.run(port=5002)` |

---

## Environment Variables (`.env`)

```env
# Optional overrides
FLASK_ENV=development
TROCR_CHECKPOINT=microsoft/trocr-base-handwritten
SBERT_CHECKPOINT=all-mpnet-base-v2
CROSS_ENC_CHECKPOINT=cross-encoder/stsb-roberta-base
```
