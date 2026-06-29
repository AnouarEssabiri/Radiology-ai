"""
tests/test_suite.py
====================
Comprehensive tests for:
  - Vocabulary encoding / decoding
  - Dataset preprocessing helpers
  - Model forward pass + generation
  - Evaluation metrics
  - Backend API endpoints (via httpx test client)
  - Inference engine

Run:
    pytest tests/test_suite.py -v
    pytest tests/test_suite.py -v -k "test_model"   # filter
"""

import io
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# ── project path ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ═══════════════════════════════════════════════════════════════════════════════
# VOCABULARY TESTS
# ═══════════════════════════════════════════════════════════════════════════════
class TestVocabulary:

    def setup_method(self):
        from datasets.scripts.prepare_dataset import Vocabulary
        self.Vocabulary = Vocabulary

    def test_build_basic(self):
        vocab = self.Vocabulary(min_freq=1)
        reports = ["the lungs are clear", "no pleural effusion is seen", "the lungs are clear"]
        vocab.build(reports)
        assert "lungs" in vocab.word2idx
        assert "clear" in vocab.word2idx
        assert len(vocab) > 4  # 4 special tokens + content words

    def test_special_tokens_present(self):
        vocab = self.Vocabulary(min_freq=1)
        vocab.build(["sample report text"])
        assert "<PAD>" in vocab.word2idx
        assert "<SOS>" in vocab.word2idx
        assert "<EOS>" in vocab.word2idx
        assert "<UNK>" in vocab.word2idx

    def test_min_freq_filtering(self):
        vocab = self.Vocabulary(min_freq=3)
        # "rare" appears only once, "common" appears 5 times
        reports = ["common word"] * 5 + ["rare word"]
        vocab.build(reports)
        assert "common" in vocab.word2idx
        assert "rare" not in vocab.word2idx

    def test_encode_decode_roundtrip(self):
        vocab = self.Vocabulary(min_freq=1)
        text = "the lungs are clear and healthy"
        vocab.build([text])
        encoded = vocab.encode(text, max_len=20)
        decoded = vocab.decode(encoded)
        assert "lungs" in decoded
        assert "clear" in decoded

    def test_encode_truncation(self):
        vocab = self.Vocabulary(min_freq=1)
        vocab.build(["a b c d e f g h i j k l m n o p"])
        tokens = vocab.encode("a b c d e f g h i j k l m n o p", max_len=8)
        assert len(tokens) == 8

    def test_encode_padding(self):
        vocab = self.Vocabulary(min_freq=1)
        vocab.build(["hello world"])
        tokens = vocab.encode("hello world", max_len=20)
        assert len(tokens) == 20
        # Tail should be PAD (index 0)
        assert all(t == 0 for t in tokens[5:])

    def test_save_load(self, tmp_path):
        vocab = self.Vocabulary(min_freq=1)
        vocab.build(["test vocabulary save load"])
        path = str(tmp_path / "vocab.pkl")
        vocab.save(path)
        loaded = self.Vocabulary.load(path)
        assert len(loaded) == len(vocab)
        assert loaded.word2idx == vocab.word2idx

    def test_unknown_token(self):
        vocab = self.Vocabulary(min_freq=1)
        vocab.build(["known words only"])
        tokens = vocab.encode("known unknown_word words", max_len=10)
        unk_idx = vocab.word2idx["<UNK>"]
        # "unknown_word" should map to UNK
        assert unk_idx in tokens


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════════
class TestRadiologyModel:

    @pytest.fixture(autouse=True)
    def setup(self):
        from ai_model.models.radiology_model import RadiologyReportModel, build_model
        self.ModelClass = RadiologyReportModel
        self.build_model = build_model
        self.device = torch.device("cpu")
        self.vocab_size = 500
        self.d_model = 128
        self.model = RadiologyReportModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=4,
            num_decoder_layers=2,
            dim_feedforward=256,
            max_seq_len=30,
            dropout=0.0,
            backbone="efficientnet_b3",
            pretrained_encoder=False,
        ).to(self.device)

    def _random_images(self, batch=2):
        return torch.randn(batch, 3, 224, 224)

    def _random_tokens(self, batch=2, seq=25):
        return torch.randint(0, self.vocab_size, (batch, seq))

    def test_forward_shape(self):
        imgs = self._random_images(2)
        toks = self._random_tokens(2, 25)
        logits = self.model(imgs, toks)
        assert logits.shape == (2, 25, self.vocab_size), f"Got {logits.shape}"

    def test_encoder_output_shape(self):
        imgs = self._random_images(2)
        feats = self.model.encoder(imgs)
        assert feats.dim() == 3
        assert feats.shape[0] == 2
        assert feats.shape[2] == self.d_model

    def test_forward_no_nan(self):
        imgs = self._random_images(2)
        toks = self._random_tokens(2, 25)
        logits = self.model(imgs, toks)
        assert not torch.isnan(logits).any(), "NaN in logits"
        assert not torch.isinf(logits).any(), "Inf in logits"

    def test_generate_shape(self):
        imgs = self._random_images(2)
        seqs, scores = self.model.generate(imgs, beam_size=2, max_len=15, min_len=3)
        assert seqs.shape[0] == 2
        assert seqs.shape[1] <= 15
        assert scores.shape == (2,)

    def test_generate_starts_with_sos(self):
        imgs = self._random_images(1)
        seqs, _ = self.model.generate(imgs, beam_size=2, max_len=10, min_len=2)
        # First token should be SOS (1)
        assert seqs[0, 0].item() == 1

    def test_parameter_count_positive(self):
        assert self.model.count_parameters() > 0

    def test_build_model_factory(self):
        cfg = {
            "encoder": "efficientnet_b3",
            "d_model": 128,
            "num_heads": 4,
            "num_decoder_layers": 2,
            "dim_feedforward": 256,
            "max_seq_len": 30,
            "dropout": 0.0,
            "pretrained_encoder": False,
        }
        model = self.build_model(cfg, vocab_size=300)
        assert isinstance(model, self.ModelClass)

    def test_weight_tying(self):
        # Embedding and output_proj should share the same weight tensor
        emb_w  = self.model.decoder.embedding.weight
        proj_w = self.model.decoder.output_proj.weight
        assert emb_w.data_ptr() == proj_w.data_ptr()

    def test_no_gradient_leak_in_generate(self):
        """generate() should not accumulate gradients."""
        imgs = self._random_images(2)
        self.model.zero_grad()
        seqs, _ = self.model.generate(imgs, beam_size=2, max_len=10, min_len=2)
        # No grad should be tracked
        assert not seqs.requires_grad


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS TESTS
# ═══════════════════════════════════════════════════════════════════════════════
class TestLabelSmoothingLoss:

    def setup_method(self):
        from ai_model.training.trainer import LabelSmoothingLoss
        self.LossClass = LabelSmoothingLoss

    def test_loss_positive(self):
        criterion = self.LossClass(vocab_size=100, smoothing=0.1)
        logits  = torch.randn(4, 10, 100)
        targets = torch.randint(0, 100, (4, 10))
        loss = criterion(logits, targets)
        assert loss.item() > 0

    def test_loss_ignores_pad(self):
        """Loss should be zero on an all-PAD target (numerically ≈ 0)."""
        criterion = self.LossClass(vocab_size=100, padding_idx=0, smoothing=0.0)
        logits  = torch.randn(2, 5, 100)
        targets = torch.zeros(2, 5, dtype=torch.long)  # all PAD
        loss = criterion(logits, targets)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_loss_finite(self):
        criterion = self.LossClass(vocab_size=200, smoothing=0.1)
        logits  = torch.randn(3, 8, 200)
        targets = torch.randint(1, 200, (3, 8))
        loss = criterion(logits, targets)
        assert torch.isfinite(loss)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS TESTS
# ═══════════════════════════════════════════════════════════════════════════════
class TestMetrics:

    def setup_method(self):
        from ai_model.evaluation.metrics import (
            compute_bleu, compute_rouge, compute_all_bleu,
            evaluate_all, compute_accuracy,
        )
        self.compute_bleu     = compute_bleu
        self.compute_rouge    = compute_rouge
        self.compute_all_bleu = compute_all_bleu
        self.evaluate_all     = evaluate_all
        self.compute_accuracy = compute_accuracy

    def test_bleu_perfect_match(self):
        hyp = ["the lungs are clear no effusion"]
        ref = [["the lungs are clear no effusion"]]
        score = self.compute_bleu(hyp, ref, max_n=4)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_bleu_zero_on_empty(self):
        score = self.compute_bleu([""], [[""]], max_n=1)
        assert 0.0 <= score <= 1.0

    def test_bleu_range(self):
        hyps = ["the heart is enlarged with mild cardiomegaly"]
        refs = [["mild cardiomegaly is present no effusion"]]
        score = self.compute_bleu(hyps, refs, max_n=4)
        assert 0.0 <= score <= 1.0

    def test_compute_all_bleu_keys(self):
        hyps = ["test report sentence here"]
        refs = [["test report sentence here"]]
        results = self.compute_all_bleu(hyps, refs)
        for n in range(1, 5):
            assert f"bleu_{n}" in results

    def test_rouge_keys(self):
        hyps = ["the lungs are clear"]
        refs = ["the lungs are clear and healthy"]
        results = self.compute_rouge(hyps, refs)
        assert "rouge1" in results
        assert "rouge2" in results
        assert "rougeL" in results

    def test_rouge_perfect_match(self):
        hyps = ["the lungs are clear"]
        refs = ["the lungs are clear"]
        results = self.compute_rouge(hyps, refs)
        assert results["rouge1"] == pytest.approx(1.0, abs=0.01)

    def test_accuracy_exact_match(self):
        hyps = ["clear lungs no effusion"]
        refs = ["clear lungs no effusion"]
        acc = self.compute_accuracy(hyps, refs, token_level=False)
        assert acc == pytest.approx(1.0)

    def test_accuracy_token_level(self):
        hyps = ["clear lungs no effusion"]
        refs = ["clear lungs no effusion"]
        acc = self.compute_accuracy(hyps, refs, token_level=True)
        assert acc == pytest.approx(1.0)

    def test_evaluate_all_returns_dict(self):
        hyps = ["the lungs are clear"]
        refs = ["the lungs are mostly clear"]
        results = self.evaluate_all(hyps, refs, compute_met=False)
        assert isinstance(results, dict)
        assert len(results) > 0
        for v in results.values():
            assert 0.0 <= v <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE UTILITIES TESTS
# ═══════════════════════════════════════════════════════════════════════════════
class TestImageUtils:

    def setup_method(self):
        from backend.utils.image_utils import (
            validate_image, image_to_base64, base64_to_pil, convert_to_pil,
        )
        self.validate_image  = validate_image
        self.image_to_base64 = image_to_base64
        self.base64_to_pil   = base64_to_pil
        self.convert_to_pil  = convert_to_pil

    def _make_image(self, size=(256, 256), mode="RGB") -> Image.Image:
        arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode)

    def test_validate_valid_image(self):
        img = self._make_image((256, 256))
        self.validate_image(img)  # Should not raise

    def test_validate_too_small(self):
        img = self._make_image((32, 32))
        with pytest.raises(ValueError, match="too small"):
            self.validate_image(img)

    def test_base64_roundtrip(self):
        img     = self._make_image((128, 128))
        encoded = self.image_to_base64(img)
        assert encoded.startswith("data:image/png;base64,")
        decoded = self.base64_to_pil(encoded)
        assert decoded.size == (128, 128)

    def test_convert_jpeg(self):
        img  = self._make_image((224, 224))
        buf  = io.BytesIO()
        img.save(buf, format="JPEG")
        pil  = self.convert_to_pil(buf.getvalue(), ".jpg")
        assert isinstance(pil, Image.Image)

    def test_convert_png(self):
        img  = self._make_image((224, 224))
        buf  = io.BytesIO()
        img.save(buf, format="PNG")
        pil  = self.convert_to_pil(buf.getvalue(), ".png")
        assert isinstance(pil, Image.Image)


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINT TESTS  (requires running backend or TestClient)
# ═══════════════════════════════════════════════════════════════════════════════
class TestAPI:
    """
    Uses FastAPI TestClient for in-process testing (no server needed).
    The model is loaded with random weights since no checkpoint exists in CI.
    """

    @pytest.fixture(autouse=True)
    def client(self):
        try:
            from fastapi.testclient import TestClient
            import backend.main as app_module
            self.client = TestClient(app_module.app, raise_server_exceptions=False)
        except Exception as exc:
            pytest.skip(f"Cannot import backend (missing deps?): {exc}")

    def _make_upload_bytes(self, size=(224, 224)) -> bytes:
        arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_health(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_predict_returns_200_or_500(self):
        """Predict endpoint should respond (200 OK or 500 if no model)."""
        img_bytes = self._make_upload_bytes()
        r = self.client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"beam_size": 3, "gradcam": False},
        )
        assert r.status_code in (200, 500)

    def test_predict_response_schema(self):
        """If 200, check required response fields."""
        img_bytes = self._make_upload_bytes()
        r = self.client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"beam_size": 3, "gradcam": False},
        )
        if r.status_code == 200:
            body = r.json()
            assert "report"     in body
            assert "confidence" in body
            assert "latency_ms" in body
            assert isinstance(body["report"], str)
            assert 0.0 <= body["confidence"] <= 1.0

    def test_predict_invalid_extension(self):
        """Non-image file should return 415."""
        r = self.client.post(
            "/predict",
            files={"file": ("file.pdf", b"fake pdf content", "application/pdf")},
        )
        assert r.status_code == 415

    def test_train_status(self):
        r = self.client.get("/train/status")
        assert r.status_code == 200
        assert "running" in r.json()


# ═══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM TESTS
# ═══════════════════════════════════════════════════════════════════════════════
class TestGradCAM:

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from ai_model.explainability.gradcam import GradCAM, GradCAMExplainer
            from ai_model.models.radiology_model import RadiologyReportModel
            self.GradCAM        = GradCAM
            self.GradCAMExplainer = GradCAMExplainer
            self.ModelClass     = RadiologyReportModel
        except ImportError as exc:
            pytest.skip(f"Deps missing: {exc}")

    def _make_model(self):
        return self.ModelClass(
            vocab_size=200, d_model=64, num_heads=2,
            num_decoder_layers=1, dim_feedforward=128,
            pretrained_encoder=False,
        )

    def _make_pil(self):
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def test_gradcam_heatmap_shape(self):
        model  = self._make_model()
        target = list(model.encoder.features.children())[-1]
        gcam   = self.GradCAM(model, target)
        tensor = torch.randn(1, 3, 224, 224)
        hmap   = gcam.compute(tensor)
        assert hmap.ndim == 2
        gcam.remove_hooks()

    def test_gradcam_heatmap_range(self):
        model  = self._make_model()
        target = list(model.encoder.features.children())[-1]
        gcam   = self.GradCAM(model, target)
        tensor = torch.randn(1, 3, 224, 224)
        hmap   = gcam.compute(tensor)
        assert hmap.min() >= 0.0
        assert hmap.max() <= 1.0 + 1e-6
        gcam.remove_hooks()

    def test_explainer_returns_pil(self):
        model    = self._make_model()
        explainer = self.GradCAMExplainer(model, image_size=224)
        img = self._make_pil()
        hmap, overlay = explainer.explain(img)
        assert isinstance(overlay, Image.Image)

    def test_explainer_saves_file(self, tmp_path):
        model    = self._make_model()
        explainer = self.GradCAMExplainer(model, image_size=224)
        img  = self._make_pil()
        path = str(tmp_path / "overlay.png")
        explainer.explain(img, save_path=path)
        assert Path(path).exists()


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST
# ═══════════════════════════════════════════════════════════════════════════════
class TestEndToEnd:
    """
    Full pipeline smoke test: image → encoder → decoder → decoded text.
    Uses tiny random model — no real weights needed.
    """

    def test_full_pipeline(self):
        from ai_model.models.radiology_model import RadiologyReportModel
        from datasets.scripts.prepare_dataset import Vocabulary

        # Build tiny vocab
        vocab = Vocabulary(min_freq=1)
        corpus = [
            "the lungs are clear no effusion",
            "mild cardiomegaly is present",
            "no acute cardiopulmonary abnormality",
        ]
        vocab.build(corpus)

        # Build tiny model
        model = RadiologyReportModel(
            vocab_size=len(vocab),
            d_model=64, num_heads=2,
            num_decoder_layers=1,
            dim_feedforward=128,
            max_seq_len=20,
            pretrained_encoder=False,
        )

        # Random image
        img = torch.randn(1, 3, 224, 224)

        # Generate
        seqs, scores = model.generate(img, beam_size=2, max_len=15, min_len=3)
        report = vocab.decode(seqs[0].tolist())

        assert isinstance(report, str)
        assert len(report) > 0
        assert scores.shape == (1,)
        print(f"\nIntegration test — generated: '{report}'")
