"""
ai_model/models/radiology_model.py
=====================================
EfficientNet-B3 visual encoder + Transformer decoder for radiology report generation.

Architecture Overview:
    Image → EfficientNet-B3 → feature map → linear projection → d_model
    Tokens → embedding → Transformer Decoder → linear → vocab logits

The decoder uses masked multi-head self-attention + cross-attention over
the visual feature tokens (standard image-captioning paradigm).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ─── Positional Encoding ─────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─── Visual Encoder ───────────────────────────────────────────────────────────
class VisualEncoder(nn.Module):
    """
    EfficientNet-B3 backbone (ImageNet pre-trained).
    Outputs a sequence of spatial feature vectors for cross-attention.
    """

    BACKBONE_OUT_CHANNELS = {
        "efficientnet_b0": 1280,
        "efficientnet_b1": 1280,
        "efficientnet_b2": 1408,
        "efficientnet_b3": 1536,
        "efficientnet_b4": 1792,
        "resnet50":         2048,
        "resnet101":        2048,
    }

    def __init__(
        self,
        backbone:   str  = "efficientnet_b3",
        d_model:    int  = 512,
        pretrained: bool = True,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.backbone_name = backbone

        # ── Load backbone ──────────────────────────────────────────────────
        weights_arg = "DEFAULT" if pretrained else None

        if backbone.startswith("efficientnet"):
            try:
                weights_cls = getattr(models, f"EfficientNet_B3_Weights", None)
                w = (weights_cls.DEFAULT if (pretrained and weights_cls) else None)
                base = getattr(models, backbone)(weights=w)
            except Exception:
                # Network unavailable — fall back to random weights
                base = getattr(models, backbone)(weights=None)
            # Keep only the feature extractor; drop the classifier
            self.features = base.features
            in_channels = self.BACKBONE_OUT_CHANNELS[backbone]

        elif backbone.startswith("resnet"):
            try:
                base = getattr(models, backbone)(
                    weights=("DEFAULT" if pretrained else None)
                )
            except Exception:
                base = getattr(models, backbone)(weights=None)
            # Remove avgpool + fc
            self.features = nn.Sequential(*list(base.children())[:-2])
            in_channels = self.BACKBONE_OUT_CHANNELS[backbone]

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ── Projection to d_model ────────────────────────────────────────
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )
        self.pos_enc = PositionalEncoding(d_model, max_len=49 * 2, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            features: (B, S, d_model)  where S = H'*W' spatial tokens
        """
        feat = self.features(images)          # (B, C, H', W')
        feat = self.proj(feat)                # (B, d_model, H', W')
        B, D, H, W = feat.shape
        feat = feat.flatten(2).permute(0, 2, 1)  # (B, S, d_model)
        feat = self.pos_enc(feat)
        return feat


# ─── Report Decoder ──────────────────────────────────────────────────────────
class ReportDecoder(nn.Module):
    """
    Standard Transformer decoder.
    Cross-attends over visual encoder output to generate report tokens.
    """

    def __init__(
        self,
        vocab_size:         int,
        d_model:            int   = 512,
        num_heads:          int   = 8,
        num_layers:         int   = 6,
        dim_feedforward:    int   = 2048,
        max_seq_len:        int   = 150,
        dropout:            float = 0.1,
        label_smoothing:    float = 0.1,
    ):
        super().__init__()
        self.d_model     = d_model
        self.vocab_size  = vocab_size
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len=max_seq_len + 1, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,         # Pre-LN for training stability
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.output_proj = nn.Linear(d_model, vocab_size)
        self.dropout     = nn.Dropout(dropout)

        # Weight tying: embedding weights shared with output projection
        self.output_proj.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask (True = masked / ignored)."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(
        self,
        tokens:         torch.Tensor,   # (B, T)  target tokens (teacher-forced)
        encoder_output: torch.Tensor,   # (B, S, d_model)  visual features
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, S)
    ) -> torch.Tensor:
        """
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = tokens.shape
        tgt_mask    = self._causal_mask(T, tokens.device)
        tgt_pad_mask = tokens.eq(0)   # PAD_IDX = 0

        emb = self.embedding(tokens) * math.sqrt(self.d_model)
        emb = self.pos_enc(emb)
        emb = self.dropout(emb)

        out = self.transformer_decoder(
            tgt=emb,
            memory=encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.output_proj(out)   # (B, T, vocab_size)
        return logits


# ─── Full Model ──────────────────────────────────────────────────────────────
class RadiologyReportModel(nn.Module):
    """
    End-to-end model: visual encoder + language decoder.

    Forward pass (training):
        images  → encoder → visual features
        tokens  → decoder (teacher-forced) → logits

    Inference:
        images → encoder → beam_search() → generated token ids
    """

    def __init__(
        self,
        vocab_size:          int,
        d_model:             int   = 512,
        num_heads:           int   = 8,
        num_decoder_layers:  int   = 6,
        dim_feedforward:     int   = 2048,
        max_seq_len:         int   = 150,
        dropout:             float = 0.1,
        backbone:            str   = "efficientnet_b3",
        pretrained_encoder:  bool  = True,
        sos_idx:             int   = 1,
        eos_idx:             int   = 2,
        pad_idx:             int   = 0,
    ):
        super().__init__()
        self.sos_idx     = sos_idx
        self.eos_idx     = eos_idx
        self.pad_idx     = pad_idx
        self.max_seq_len = max_seq_len

        self.encoder = VisualEncoder(
            backbone=backbone,
            d_model=d_model,
            pretrained=pretrained_encoder,
            dropout=dropout,
        )
        self.decoder = ReportDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

    def forward(
        self,
        images: torch.Tensor,   # (B, 3, H, W)
        tokens: torch.Tensor,   # (B, T)  teacher-forced target
    ) -> torch.Tensor:
        """Returns logits (B, T, vocab_size)."""
        visual_features = self.encoder(images)
        logits = self.decoder(tokens, visual_features)
        return logits

    @torch.no_grad()
    def generate(
        self,
        images:    torch.Tensor,
        beam_size: int = 5,
        max_len:   int = 150,
        min_len:   int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Beam search decoding.

        Returns:
            best_sequences : (B, max_len)   token ids
            scores         : (B,)           log-prob scores
        """
        self.eval()
        device = images.device
        B      = images.size(0)

        visual_features = self.encoder(images)   # (B, S, d_model)

        # Expand for beam search: (B * beam_size, S, d_model)
        visual_features = visual_features.unsqueeze(1) \
            .expand(-1, beam_size, -1, -1) \
            .reshape(B * beam_size, -1, visual_features.size(-1))

        # Sequences: (B, beam_size, seq_len)
        sequences  = torch.full((B, beam_size, 1), self.sos_idx,
                                dtype=torch.long, device=device)
        scores     = torch.zeros(B, beam_size, device=device)
        finished   = torch.zeros(B, beam_size, dtype=torch.bool, device=device)

        for step in range(max_len - 1):
            # Reshape for decoder: (B*beam_size, current_len)
            curr_len   = sequences.size(2)
            tgt        = sequences.reshape(B * beam_size, curr_len)
            logits     = self.decoder(tgt, visual_features)          # (B*BS, curr_len, V)
            next_logits = logits[:, -1, :]                           # (B*BS, V)
            log_probs   = F.log_softmax(next_logits, dim=-1)         # (B*BS, V)
            log_probs   = log_probs.reshape(B, beam_size, -1)        # (B, BS, V)

            # Penalise EOS before min_len
            if step < min_len:
                log_probs[:, :, self.eos_idx] = -1e9

            # Accumulate scores
            total_scores = (scores.unsqueeze(-1) + log_probs)        # (B, BS, V)
            total_scores = total_scores.reshape(B, -1)               # (B, BS*V)

            # Select top-k
            top_scores, top_idx = total_scores.topk(beam_size, dim=-1)
            beam_idx  = top_idx // log_probs.size(-1)
            token_idx = top_idx  % log_probs.size(-1)

            # Re-order sequences
            sequences = sequences[
                torch.arange(B, device=device).unsqueeze(-1),
                beam_idx
            ]
            sequences = torch.cat(
                [sequences, token_idx.unsqueeze(-1)], dim=-1
            )
            scores = top_scores

            # Mark finished beams
            finished = finished[
                torch.arange(B, device=device).unsqueeze(-1), beam_idx
            ]
            finished |= (token_idx == self.eos_idx)

            if finished.all():
                break

        # Return best beam
        best_seq   = sequences[:, 0, :]          # (B, seq_len)
        best_score = scores[:, 0]                # (B,)
        return best_seq, best_score

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── ViT-based variant ────────────────────────────────────────────────────────
class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder using torchvision's ViT-B/16.
    Drop-in replacement for VisualEncoder.
    """

    def __init__(self, d_model: int = 512, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        base = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
        self.patch_embed   = base.conv_proj
        self.encoder       = base.encoder
        self.class_token   = base.class_token
        vit_hidden         = 768
        self.proj          = nn.Linear(vit_hidden, d_model)
        self.pos_enc       = PositionalEncoding(d_model, max_len=197, dropout=dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, 3, 224, 224)
        B  = images.size(0)
        x  = self.patch_embed(images)                      # (B, 768, 14, 14)
        x  = x.flatten(2).permute(0, 2, 1)                # (B, 196, 768)
        cls = self.class_token.expand(B, -1, -1)           # (B, 1, 768)
        x  = torch.cat([cls, x], dim=1)                    # (B, 197, 768)
        x  = self.encoder(x)                               # (B, 197, 768)
        x  = self.proj(x)                                  # (B, 197, d_model)
        x  = self.pos_enc(x)
        return x


# ─── Factory ─────────────────────────────────────────────────────────────────
def build_model(cfg: dict, vocab_size: int) -> RadiologyReportModel:
    """
    Instantiate a RadiologyReportModel from a config dict.

    cfg keys mirror config/config.yaml → model section.
    """
    return RadiologyReportModel(
        vocab_size          = vocab_size,
        d_model             = cfg.get("d_model", 512),
        num_heads           = cfg.get("num_heads", 8),
        num_decoder_layers  = cfg.get("num_decoder_layers", 6),
        dim_feedforward     = cfg.get("dim_feedforward", 2048),
        max_seq_len         = cfg.get("max_seq_len", 150),
        dropout             = cfg.get("dropout", 0.1),
        backbone            = cfg.get("encoder", "efficientnet_b3"),
        pretrained_encoder  = cfg.get("pretrained_encoder", True),
    )


if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = RadiologyReportModel(vocab_size=3000).to(device)

    imgs   = torch.randn(2, 3, 224, 224, device=device)
    toks   = torch.randint(0, 3000, (2, 50), device=device)
    logits = model(imgs, toks)

    print(f"Logits shape    : {logits.shape}")       # (2, 50, 3000)
    print(f"Parameters      : {model.count_parameters():,}")
    seqs, sc = model.generate(imgs, beam_size=3, max_len=30)
    print(f"Generated shape : {seqs.shape}")          # (2, ≤30)
    print(f"Beam scores     : {sc}")
