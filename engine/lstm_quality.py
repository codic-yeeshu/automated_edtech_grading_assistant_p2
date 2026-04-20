"""
BiLSTM Answer Quality Scorer with Scaled Dot-Product Attention.

Architecture:
  Input text → sentence-level SBERT embeddings → BiLSTM (2 layers) →
  Attention pooling → MLP head → quality score ∈ [0, 1]

The model operates over sentence sequences (not token sequences) to keep
inference fast on CPU without losing long-range structure.

Regularization:
  - Dropout(0.3) after projection and before classifier
  - Orthogonal init for LSTM recurrent weights (reduces vanishing gradients)
  - Xavier uniform for input-to-hidden weights
  - LayerNorm after projection
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

# ─── Attention layer ──────────────────────────────────────────────────────────

class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention over LSTM hidden states.

    Computes a weighted sum of all time steps, where weights are derived
    from learned query vector q attending over LSTM outputs.

    Attention formula: α = softmax(q · H^T / √d), context = α · H
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** 0.5

    def forward(self, lstm_output: torch.Tensor):
        """
        Args:
            lstm_output: [batch, seq_len, hidden_dim]
        Returns:
            context:     [batch, hidden_dim]
            attn_weights:[batch, seq_len]
        """
        Q = self.query(lstm_output)                         # [B, T, H]
        K = self.key(lstm_output)                           # [B, T, H]
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [B, T, T]
        attn_weights = F.softmax(scores.mean(dim=1), dim=-1)    # [B, T]
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)  # [B, H]
        return context, attn_weights


# ─── BiLSTM model ─────────────────────────────────────────────────────────────

class BiLSTMScorer(nn.Module):
    """
    Bidirectional LSTM with attention for answer quality scoring.

    Args:
        input_dim:  Embedding dimension of input (SBERT = 768)
        proj_dim:   Projection size before LSTM (reduces compute)
        hidden_dim: LSTM hidden units per direction
        num_layers: Number of stacked LSTM layers
        dropout:    Dropout probability
    """

    def __init__(
        self,
        input_dim: int = 768,
        proj_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Projection: reduce 768-dim SBERT embeddings to proj_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Bi-directional LSTM (output dim = hidden_dim * 2)
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = ScaledDotProductAttention(hidden_dim * 2)
        self.dropout   = nn.Dropout(dropout)

        # MLP classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal init for LSTM recurrent weights (aids gradient flow).
        Xavier uniform for input-to-hidden weights (better than random Gaussian).
        """
        for name, param in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)
            elif "linear" in name.lower() or "classifier" in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param.data)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch, seq_len, input_dim]  (sentence embeddings)
        Returns:
            score:       [batch, 1]
            attn_weights:[batch, seq_len]
        """
        projected = self.projection(x)          # [B, T, proj_dim]
        lstm_out, _ = self.lstm(projected)       # [B, T, hidden*2]
        context, attn_w = self.attention(lstm_out)  # [B, hidden*2], [B, T]
        context = self.dropout(context)
        score = self.classifier(context)         # [B, 1]
        return score, attn_w


# ─── High-level scorer ────────────────────────────────────────────────────────

class NeuralQualityScorer:
    """
    End-to-end answer quality scorer.

    Pipeline:
      text → sentence split → SBERT sentence embeddings →
      BiLSTM + attention → relative quality score [0, 1]

    The score reflects how semantically complete the student answer is
    relative to the teacher's reference answer.
    """

    MAX_SENTENCES = 10

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = BiLSTMScorer().to(self.device)
        self.model.eval()
        self._sbert  = None

    @property
    def sbert(self):
        if self._sbert is None:
            from engine.model_registry import ModelRegistry
            self._sbert = ModelRegistry.sbert()
        return self._sbert

    def _sentence_embeddings(self, text: str) -> torch.Tensor:
        """
        Split text into sentences, embed each with SBERT.
        Returns: [1, num_sentences, 768]
        """
        sentences = [s.strip() for s in re.split(r"[.!?]+", text.strip()) if s.strip()]
        sentences = sentences[: self.MAX_SENTENCES] or [text.strip()]
        emb = self.sbert.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
        return emb.unsqueeze(0).to(self.device)    # [1, T, 768]

    def score(self, student_text: str, teacher_text: str) -> float:
        """
        Compute answer quality score in [0, 1].

        Combines:
          - BiLSTM relative quality (student vs teacher self-score)
          - Sentence-level cosine similarity (direct embedding comparison)
        """
        if not student_text.strip() or not teacher_text.strip():
            return 0.0

        with torch.no_grad():
            s_emb = self._sentence_embeddings(student_text)   # [1, Ts, 768]
            t_emb = self._sentence_embeddings(teacher_text)   # [1, Tt, 768]

            s_score, _ = self.model(s_emb)   # [1, 1]
            t_score, _ = self.model(t_emb)   # [1, 1]

            s_val = s_score.item()
            t_val = max(t_score.item(), 1e-6)

            # Mean-pool embeddings for cosine similarity
            s_flat = s_emb.mean(dim=1)   # [1, 768]
            t_flat = t_emb.mean(dim=1)   # [1, 768]
            cosine = F.cosine_similarity(s_flat, t_flat).item()

            # Blend relative LSTM quality (40%) + embedding cosine (60%)
            quality = 0.4 * min(s_val / t_val, 1.0) + 0.6 * max(0.0, cosine)
            return round(float(min(max(quality, 0.0), 1.0)), 4)
