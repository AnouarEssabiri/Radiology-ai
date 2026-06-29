"""
ai_model/inference.py
========================
High-level inference engine used by the FastAPI backend.
Loads the trained model once and exposes a predict() method.
"""

import io
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


class RadiologyInferenceEngine:
    """
    Singleton-style inference engine.

    Usage:
        engine = RadiologyInferenceEngine.from_config("config/config.yaml")
        result = engine.predict(pil_image)
    """

    _instance: Optional["RadiologyInferenceEngine"] = None

    def __init__(
        self,
        model_path:  str,
        vocab_path:  str,
        cfg:         dict,
        device:      Optional[torch.device] = None,
    ):
        self.cfg    = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Inference device: %s", self.device)

        # ── Vocabulary ────────────────────────────────────────────────────
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        logger.info("Vocab loaded: %d tokens", len(self.vocab))

        # ── Model ─────────────────────────────────────────────────────────
        from ai_model.models.radiology_model import build_model

        self.model = build_model(cfg["model"], vocab_size=len(self.vocab))
        self._load_weights(model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded from %s", model_path)

        # ── LLM Integration ───────────────────────────────────────────────
        self.use_llm = cfg.get("llm", {}).get("enabled", False)
        self.llm_generator = None
        if self.use_llm:
            try:
                from ai_model.llm_integration import LLMReportGenerator
                self.llm_generator = LLMReportGenerator.from_config(cfg)
                logger.info("LLM integration enabled.")
            except Exception as exc:
                logger.warning("LLM init failed: %s", exc)
                self.use_llm = False

        # ── Grad-CAM ──────────────────────────────────────────────────────
        try:
            from ai_model.explainability.gradcam import GradCAMExplainer
            self.explainer = GradCAMExplainer(
                self.model,
                image_size=cfg["dataset"]["image_size"],
                alpha=cfg["explainability"].get("alpha", 0.5),
                device=self.device,
            )
        except Exception as exc:
            logger.warning("Grad-CAM init failed: %s", exc)
            self.explainer = None

        # ── Image transform ────────────────────────────────────────────────
        sz = cfg["dataset"]["image_size"]
        self.transform = transforms.Compose([
            transforms.Resize((sz, sz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _load_weights(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning("No checkpoint found at %s – running with random weights.", path)
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state, strict=False)

    # ── Factory ───────────────────────────────────────────────────────────────
    @classmethod
    def from_config(cls, config_path: str = "config/config.yaml") -> "RadiologyInferenceEngine":
        if cls._instance is None:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            cls._instance = cls(
                model_path=cfg["training"]["best_model_path"],
                vocab_path=cfg["dataset"]["paths"]["vocab"],
                cfg=cfg,
            )
        return cls._instance

    # ── Core predict ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        image:           Image.Image,
        beam_size:       int  = 5,
        generate_gradcam: bool = True,
    ) -> Dict:
        """
        Run inference on a single PIL image.

        Returns:
            {
                "report":       str,           # generated radiology report
                "confidence":   float,         # normalised beam score [0,1]
                "gradcam_pil":  PIL.Image,     # Grad-CAM overlay (or None)
                "latency_ms":   float,
            }
        """
        t0 = time.perf_counter()

        # ── Preprocess ────────────────────────────────────────────────────
        image_rgb = image.convert("RGB")
        tensor    = self.transform(image_rgb).unsqueeze(0).to(self.device)  # (1,3,H,W)

        # ── Generate report ────────────────────────────────────────────────
        inf_cfg   = self.cfg.get("inference", {})
        seqs, scores = self.model.generate(
            tensor,
            beam_size=beam_size,
            max_len  =inf_cfg.get("max_len", 150),
            min_len  =inf_cfg.get("min_len", 20),
        )

        base_report = self.vocab.decode(seqs[0].cpu().tolist())
        base_report = self._post_process(base_report)

        # Use LLM to refine the report if enabled
        if self.use_llm and self.llm_generator:
            report = self.llm_generator.generate_report(base_report)
        else:
            report = base_report

        # Convert log-prob score → approximate confidence [0, 1]
        # Since scores[0] is the accumulated log-probability of the sequence,
        # we calculate the average log-probability per generated token to get
        # a length-normalized confidence score: exp(accumulated_log_prob / length)
        tokens_list = seqs[0].cpu().tolist()
        try:
            # Find the index of EOS (2) and count everything up to and including EOS
            seq_len = tokens_list.index(self.model.eos_idx) + 1
        except ValueError:
            seq_len = len(tokens_list)
        
        # We start with SOS, so the number of generated tokens is seq_len - 1
        num_generated_tokens = max(1, seq_len - 1)
        avg_log_prob = scores[0].item() / num_generated_tokens
        confidence = float(np.exp(avg_log_prob))
        confidence = min(1.0, max(0.0, confidence))

        # ── Grad-CAM ───────────────────────────────────────────────────────
        gradcam_pil = None
        if generate_gradcam and self.explainer is not None:
            try:
                _, gradcam_pil = self.explainer.explain(image_rgb)
            except Exception as exc:
                logger.warning("Grad-CAM failed: %s", exc)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "report":       report,
            "confidence":   confidence,
            "gradcam_pil":  gradcam_pil,
            "latency_ms":   round(latency_ms, 1),
        }

    @staticmethod
    def _post_process(text: str) -> str:
        """Light-touch post-processing to improve readability."""
        text = text.strip()
        # Capitalise first letter of each sentence
        sentences = text.split(". ")
        sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
        text = ". ".join(sentences)
        # Ensure trailing period
        if text and not text.endswith("."):
            text += "."
        return text

    # ── Batch inference ───────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_batch(
        self,
        images:    list,   # list of PIL Images
        beam_size: int = 5,
    ) -> list:
        """Run prediction on multiple images (no Grad-CAM)."""
        results = []
        for img in images:
            results.append(self.predict(img, beam_size=beam_size, generate_gradcam=False))
        return results

    # ── Evaluation helper ─────────────────────────────────────────────────────
    def evaluate_on_test_set(self) -> Dict:
        """Run full evaluation on the test split and return metrics."""
        from datasets.scripts.dataloader import build_dataloaders
        from ai_model.evaluation.metrics import evaluate_all

        _, _, test_dl = build_dataloaders(
            annotations_path=self.cfg["dataset"]["paths"]["annotations"],
            vocab_path       =self.cfg["dataset"]["paths"]["vocab"],
            batch_size       =16,
            num_workers      =2,
        )

        hypotheses, references = [], []
        for batch in test_dl:
            images = batch["image"].to(self.device)
            seqs, _ = self.model.generate(images, beam_size=3, max_len=80)
            for seq, ref in zip(seqs, batch["report"]):
                hypotheses.append(self.vocab.decode(seq.cpu().tolist()))
                references.append(ref)

        return evaluate_all(hypotheses, references)
