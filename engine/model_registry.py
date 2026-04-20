"""
Model Registry — lazy-loaded singleton cache for all DL models.

Models are downloaded on first use to avoid startup latency.
Supports TrOCR (handwriting OCR), SBERT bi-encoder, and RoBERTa cross-encoder.
"""

import logging
import torch

logger = logging.getLogger(__name__)

# ─── TrOCR availability guard ────────────────────────────────────────────────
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    _TROCR_AVAILABLE = True
except ImportError:
    _TROCR_AVAILABLE = False
    logger.warning("transformers not found — TrOCR unavailable, falling back to EasyOCR")

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.warning("sentence-transformers not found — DL grading modes degraded")


class ModelRegistry:
    """
    Singleton registry that lazily loads and caches all DL models.

    Usage:
        proc, model = ModelRegistry.trocr()
        sbert        = ModelRegistry.sbert()
        ce           = ModelRegistry.cross_encoder()
    """

    _trocr_processor  = None
    _trocr_model      = None
    _sbert_model      = None
    _cross_encoder    = None

    # Model identifiers (HuggingFace Hub or local)
    TROCR_CHECKPOINT      = "microsoft/trocr-base-handwritten"
    SBERT_CHECKPOINT      = "all-mpnet-base-v2"
    CROSS_ENC_CHECKPOINT  = "cross-encoder/stsb-roberta-base"

    @classmethod
    def device(cls) -> torch.device:
        """Return the best available compute device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── TrOCR ──────────────────────────────────────────────────────────────
    @classmethod
    def trocr(cls):
        """
        Load TrOCR (ViT encoder + GPT-2-style autoregressive decoder).

        Architecture:
          - Encoder: Vision Transformer (ViT-Base / ViT-Large) processes 384×384 image patches
          - Decoder: GPT-2-style transformer, generates text token by token
          - Pre-trained on IAM + SROIE + other handwriting datasets

        Returns:
            (TrOCRProcessor, VisionEncoderDecoderModel)
        """
        if not _TROCR_AVAILABLE:
            raise RuntimeError(
                "transformers library required for TrOCR. "
                "Install with: pip install transformers"
            )
        if cls._trocr_processor is None:
            logger.info("Loading TrOCR from %s …", cls.TROCR_CHECKPOINT)
            cls._trocr_processor = TrOCRProcessor.from_pretrained(cls.TROCR_CHECKPOINT)
            cls._trocr_model = VisionEncoderDecoderModel.from_pretrained(
                cls.TROCR_CHECKPOINT
            ).to(cls.device())
            cls._trocr_model.eval()
            logger.info("TrOCR loaded (device=%s)", cls.device())
        return cls._trocr_processor, cls._trocr_model

    # ── SBERT bi-encoder ───────────────────────────────────────────────────
    @classmethod
    def sbert(cls) -> "SentenceTransformer":
        """
        Load SBERT all-mpnet-base-v2 bi-encoder.

        Stronger than all-MiniLM-L6-v2 (Phase 1); ~420 MB.
        Used for sentence-level embedding and cosine similarity.
        """
        if not _ST_AVAILABLE:
            raise RuntimeError("sentence-transformers required: pip install sentence-transformers")
        if cls._sbert_model is None:
            logger.info("Loading SBERT %s …", cls.SBERT_CHECKPOINT)
            cls._sbert_model = SentenceTransformer(cls.SBERT_CHECKPOINT)
            logger.info("SBERT loaded")
        return cls._sbert_model

    # ── Cross-encoder ──────────────────────────────────────────────────────
    @classmethod
    def cross_encoder(cls) -> "CrossEncoder":
        """
        Load RoBERTa cross-encoder (stsb-roberta-base).

        Unlike bi-encoders, the cross-encoder processes (query, passage) jointly,
        allowing full cross-attention between student and teacher answers.
        Output: regression score in [0, 1] (STS-B scale).
        """
        if not _ST_AVAILABLE:
            raise RuntimeError("sentence-transformers required: pip install sentence-transformers")
        if cls._cross_encoder is None:
            logger.info("Loading cross-encoder %s …", cls.CROSS_ENC_CHECKPOINT)
            cls._cross_encoder = CrossEncoder(cls.CROSS_ENC_CHECKPOINT, max_length=512)
            logger.info("Cross-encoder loaded")
        return cls._cross_encoder

    @classmethod
    def preload_all(cls):
        """Eagerly load every model (useful to trigger downloads before first request)."""
        cls.trocr()
        cls.sbert()
        cls.cross_encoder()
        logger.info("All DL models preloaded.")
