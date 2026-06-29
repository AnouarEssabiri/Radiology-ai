"""
ai_model/training/trainer.py
==============================
Full training loop for the RadiologyReportModel.

Features:
  - Teacher-forced cross-entropy with label smoothing
  - Mixed-precision (FP16) via torch.cuda.amp
  - Cosine LR scheduler with warmup
  - Gradient clipping
  - Early stopping
  - Checkpoint save / resume
  - TensorBoard logging
  - Per-epoch BLEU-4 on validation set

Usage:
    python ai_model/training/trainer.py
    python ai_model/training/trainer.py --config config/config.yaml
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
try:
    from torch.amp import GradScaler, autocast          # PyTorch >= 2.1
except ImportError:
    from torch.cuda.amp import GradScaler, autocast     # PyTorch < 2.1
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # tensorboard optional

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_model.models.radiology_model import build_model
from ai_model.evaluation.metrics import compute_bleu
from datasets.scripts.dataloader import build_dataloaders
from datasets.scripts.prepare_dataset import Vocabulary

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("Trainer")


# ─── Loss ─────────────────────────────────────────────────────────────────────
class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with label smoothing. Ignores PAD token (index 0).
    """

    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.smoothing   = smoothing
        self.padding_idx = padding_idx
        self.vocab_size  = vocab_size

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits : (B, T, V)   targets : (B, T)
        B, T, V = logits.shape
        logits  = logits.reshape(-1, V)    # (B*T, V)
        targets = targets.reshape(-1)      # (B*T,)

        log_probs = torch.log_softmax(logits, dim=-1)

        # Build smooth target distribution
        with torch.no_grad():
            smooth_target = torch.full_like(log_probs, self.smoothing / (V - 2))
            smooth_target.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            smooth_target[:, self.padding_idx] = 0.0
            mask = targets.eq(self.padding_idx)
            smooth_target[mask] = 0.0

        loss = -(smooth_target * log_probs).sum(dim=-1)
        non_pad = (~mask).sum()
        return loss.sum() / non_pad.clamp(min=1)


# ─── Warmup Scheduler ────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    """Linear warmup then cosine decay."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, eta_min: float = 1e-6):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.eta_min      = eta_min
        self.base_lrs     = [pg["lr"] for pg in optimizer.param_groups]
        self._step        = 0

    def step(self) -> None:
        self._step += 1
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self._step <= self.warmup_steps:
                pg["lr"] = base_lr * (self._step / max(1, self.warmup_steps))
            else:
                progress = (self._step - self.warmup_steps) / max(
                    1, self.total_steps - self.warmup_steps
                )
                pg["lr"] = self.eta_min + 0.5 * (base_lr - self.eta_min) * (
                    1 + math.cos(math.pi * progress)
                )

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ─── Trainer ─────────────────────────────────────────────────────────────────
class Trainer:

    def __init__(self, cfg: dict):
        self.cfg  = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s", self.device)

        # ── Dataloaders ───────────────────────────────────────────────────
        self.train_dl, self.val_dl, self.test_dl = build_dataloaders(
            annotations_path=cfg["dataset"]["paths"]["annotations"],
            vocab_path       =cfg["dataset"]["paths"]["vocab"],
            batch_size       =cfg["training"]["batch_size"],
            image_size       =cfg["dataset"]["image_size"],
            max_seq_len      =cfg["dataset"]["max_seq_len"],
            num_workers      =cfg["dataset"]["num_workers"],
            pin_memory       =cfg["dataset"]["pin_memory"],
        )

        # ── Vocabulary (needed to decode during eval) ─────────────────────
        import pickle
        with open(cfg["dataset"]["paths"]["vocab"], "rb") as f:
            self.vocab: Vocabulary = pickle.load(f)

        vocab_size = len(self.vocab)
        logger.info("Vocabulary size: %d", vocab_size)

        # ── Model ─────────────────────────────────────────────────────────
        self.model = build_model(cfg["model"], vocab_size).to(self.device)
        logger.info("Parameters: %s", f"{self.model.count_parameters():,}")

        # ── Loss ──────────────────────────────────────────────────────────
        self.criterion = LabelSmoothingLoss(
            vocab_size  =vocab_size,
            smoothing   =cfg["training"].get("label_smoothing", 0.1),
        )

        # ── Optimizer ─────────────────────────────────────────────────────
        self.optimizer = AdamW(
            self.model.parameters(),
            lr           =cfg["training"]["learning_rate"],
            weight_decay =cfg["training"].get("weight_decay", 1e-5),
        )

        # ── Scheduler ─────────────────────────────────────────────────────
        total_steps   = cfg["training"]["epochs"] * len(self.train_dl)
        warmup_steps  = cfg["training"].get("warmup_steps", 500)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, warmup_steps, total_steps,
            eta_min=cfg["training"]["scheduler"].get("eta_min", 1e-6),
        )

        # ── Mixed precision ────────────────────────────────────────────────
        self.use_amp = cfg["training"].get("mixed_precision", True) \
                       and torch.cuda.is_available()
        try:
            self.scaler = GradScaler("cuda", enabled=self.use_amp)
        except TypeError:
            self.scaler = GradScaler(enabled=self.use_amp)

        # ── Logging ───────────────────────────────────────────────────────
        os.makedirs(cfg["logging"]["tensorboard_dir"], exist_ok=True)
        if SummaryWriter is not None:
            self.writer = SummaryWriter(cfg["logging"]["tensorboard_dir"])
        else:
            self.writer = None

        # ── Checkpointing ─────────────────────────────────────────────────
        os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
        self.best_model_path = cfg["training"]["best_model_path"]
        self.best_bleu       = -1.0
        self.patience        = cfg["training"].get("early_stopping_patience", 7)
        self.bad_epochs      = 0

        self.global_step  = 0
        self.start_epoch  = 1

    # ── Training step ─────────────────────────────────────────────────────────
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        log_interval = self.cfg["training"].get("log_interval", 50)
        clip_norm    = self.cfg["training"].get("clip_grad_norm", 1.0)

        for step, batch in enumerate(self.train_dl):
            images = batch["image"].to(self.device, non_blocking=True)
            tokens = batch["tokens"].to(self.device, non_blocking=True)

            # Decoder input  : tokens[:, :-1]  (no final EOS)
            # Target         : tokens[:, 1:]   (shifted by 1)
            dec_input = tokens[:, :-1]
            target    = tokens[:, 1:]

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda" if self.use_amp else "cpu", enabled=self.use_amp):
                logits = self.model(images, dec_input)      # (B, T-1, V)
                loss   = self.criterion(logits, target)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            self.global_step += 1

            if (step + 1) % log_interval == 0:
                avg = total_loss / (step + 1)
                logger.info(
                    "Epoch %d  Step %d/%d  loss=%.4f  lr=%.2e",
                    epoch, step + 1, len(self.train_dl), avg,
                    self.scheduler.current_lr,
                )
                if self.writer: self.writer.add_scalar("Train/loss", avg, self.global_step)
                if self.writer: self.writer.add_scalar("Train/lr", self.scheduler.current_lr, self.global_step)

        return total_loss / len(self.train_dl)

    # ── Validation step ────────────────────────────────────────────────────────
    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        hypotheses, references = [], []
        beam_size = self.cfg["inference"].get("beam_size", 3)

        for batch in self.val_dl:
            images = batch["image"].to(self.device, non_blocking=True)
            tokens = batch["tokens"].to(self.device, non_blocking=True)

            dec_input = tokens[:, :-1]
            target    = tokens[:, 1:]

            with autocast(device_type="cuda" if self.use_amp else "cpu", enabled=self.use_amp):
                logits = self.model(images, dec_input)
                loss   = self.criterion(logits, target)
            total_loss += loss.item()

            # Generate for BLEU
            gen_seqs, _ = self.model.generate(images, beam_size=beam_size, max_len=80)
            for seq, ref_tok in zip(gen_seqs, tokens):
                hyp = self.vocab.decode(seq.cpu().tolist())
                ref = self.vocab.decode(ref_tok.cpu().tolist())
                hypotheses.append(hyp)
                references.append([ref])   # BLEU expects list of refs

        avg_loss = total_loss / len(self.val_dl)
        bleu4    = compute_bleu(hypotheses, references, max_n=4)

        if self.writer: self.writer.add_scalar("Val/loss",  avg_loss, epoch)
        if self.writer: self.writer.add_scalar("Val/BLEU4", bleu4,    epoch)

        return {"loss": avg_loss, "bleu4": bleu4}

    # ── Checkpoint helpers ─────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool) -> None:
        ckpt = {
            "epoch":       epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "scaler":      self.scaler.state_dict(),
            "metrics":     metrics,
            "config":      self.cfg,
        }
        path = os.path.join(
            self.cfg["training"]["checkpoint_dir"], f"checkpoint_epoch_{epoch:03d}.pth"
        )
        torch.save(ckpt, path)
        if is_best:
            torch.save(ckpt, self.best_model_path)
            logger.info("✓ Best model saved (BLEU-4 = %.4f)", metrics["bleu4"])

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_bleu   = ckpt["metrics"].get("bleu4", -1.0)
        logger.info("Resumed from epoch %d", ckpt["epoch"])

    # ── Main train loop ────────────────────────────────────────────────────────
    def train(self) -> None:
        epochs = self.cfg["training"]["epochs"]
        logger.info("Starting training for %d epochs …", epochs)

        for epoch in range(self.start_epoch, epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            metrics    = self._val_epoch(epoch)
            elapsed    = time.time() - t0

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | BLEU-4=%.4f | %.0fs",
                epoch, epochs,
                train_loss, metrics["loss"], metrics["bleu4"], elapsed,
            )

            # ── Save checkpoint ───────────────────────────────────────────
            is_best = metrics["bleu4"] > self.best_bleu
            if is_best:
                self.best_bleu  = metrics["bleu4"]
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1

            self._save_checkpoint(epoch, metrics, is_best)

            # ── Early stopping ────────────────────────────────────────────
            if self.bad_epochs >= self.patience:
                logger.info("Early stopping triggered after %d bad epochs.", self.patience)
                break

        if self.writer: self.writer.close()
        logger.info("Training complete.  Best BLEU-4 = %.4f", self.best_bleu)


# ─── CLI ─────────────────────────────────────────────────────────────────────
def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--resume",  default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg     = load_cfg(args.config)
    trainer = Trainer(cfg)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
