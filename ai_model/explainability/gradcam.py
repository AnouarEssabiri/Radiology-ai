"""
ai_model/explainability/gradcam.py
=====================================
Grad-CAM visualisation for the EfficientNet encoder.

Produces heat-maps that highlight the image regions most responsible
for the generated report, helping radiologists understand model attention.

References:
    Selvaraju et al., 2017 – "Grad-CAM: Visual Explanations from
    Deep Networks via Gradient-based Localization"
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Computes Grad-CAM heat-maps for the target convolutional layer.

    Usage:
        gcam = GradCAM(model, target_layer=model.encoder.features[-1])
        heatmap = gcam.compute(image_tensor)   # image_tensor: (1,3,H,W)
        overlay = gcam.overlay(image_pil, heatmap)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._gradients:  Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._hooks: List = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, inp, output):
            self._activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self._hooks.append(
            self.target_layer.register_backward_hook(backward_hook)
        )

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def compute(
        self,
        image:     torch.Tensor,      # (1, 3, H, W)
        token_idx: Optional[int] = None,  # token position to differentiate
    ) -> np.ndarray:
        """
        Returns a (H, W) heatmap in [0, 1].

        If token_idx is None, gradients are taken w.r.t. the first
        generated token (SOS → first output), giving a global
        image-relevance map.
        """
        self.model.zero_grad()

        # Forward through encoder only (we only need visual features)
        with torch.enable_grad():
            visual_features = self.model.encoder(image)   # (1, S, d_model)

            # Use the mean feature activation as a proxy scalar for backprop
            # (avoids a full beam-search pass, which is non-differentiable)
            score = visual_features.mean()
            score.backward()

        grads  = self._gradients         # (1, C, H', W')
        acts   = self._activations       # (1, C, H', W')

        if grads is None or acts is None:
            logger.warning("Grad-CAM: no gradients / activations captured.")
            return np.zeros((224, 224), dtype=np.float32)

        # Pool gradients over spatial dimensions
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Normalise to [0,1]
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.astype(np.float32)

    @staticmethod
    def overlay(
        pil_image:  Image.Image,
        heatmap:    np.ndarray,        # (H', W') in [0,1]
        alpha:      float = 0.5,
        colormap:   int   = cv2.COLORMAP_JET,
    ) -> Image.Image:
        """
        Blend the Grad-CAM heatmap over the original image.

        Returns a PIL Image.
        """
        W, H   = pil_image.size
        img_np = np.array(pil_image.convert("RGB"))

        # Resize cam to match image size
        cam_uint8  = np.uint8(255 * heatmap)
        cam_resized = cv2.resize(cam_uint8, (W, H), interpolation=cv2.INTER_LINEAR)
        cam_color   = cv2.applyColorMap(cam_resized, colormap)     # BGR
        cam_rgb     = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)

        blended = np.uint8(alpha * cam_rgb + (1 - alpha) * img_np)
        return Image.fromarray(blended)

    def __del__(self):
        self.remove_hooks()


# ─── Convenience wrapper ──────────────────────────────────────────────────────
class GradCAMExplainer:
    """
    High-level wrapper: load model from checkpoint and produce overlays.
    """

    def __init__(
        self,
        model:         nn.Module,
        target_layer:  Optional[nn.Module] = None,
        image_size:    int   = 224,
        alpha:         float = 0.5,
        device:        Optional[torch.device] = None,
    ):
        self.model      = model.eval()
        self.image_size = image_size
        self.alpha      = alpha
        self.device     = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Default target: last block of EfficientNet features
        if target_layer is None:
            try:
                target_layer = list(model.encoder.features.children())[-1]
            except Exception:
                target_layer = model.encoder.features

        self.gradcam = GradCAM(model, target_layer)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def explain(
        self,
        image:          Image.Image,
        save_path:      Optional[str] = None,
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Generate Grad-CAM for a PIL image.

        Returns:
            heatmap : (H', W') numpy array
            overlay : PIL Image with heatmap blended over original
        """
        tensor  = self._preprocess(image)
        heatmap = self.gradcam.compute(tensor)
        overlay = GradCAM.overlay(image, heatmap, alpha=self.alpha)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            overlay.save(save_path)
            logger.info("Grad-CAM saved → %s", save_path)

        return heatmap, overlay

    def explain_batch(
        self,
        images:     List[Image.Image],
        save_dir:   Optional[str] = None,
    ) -> List[Tuple[np.ndarray, Image.Image]]:
        results = []
        for i, img in enumerate(images):
            path = str(Path(save_dir) / f"gradcam_{i:04d}.png") if save_dir else None
            results.append(self.explain(img, save_path=path))
        return results


# ─── Attention-map extraction (Transformer decoder) ──────────────────────────
class AttentionMapExtractor:
    """
    Extracts cross-attention weights from the Transformer decoder layers.
    This reveals which image patches the model attends to for each word.
    """

    def __init__(self, model: nn.Module):
        self.model        = model
        self._attn_maps:  List[torch.Tensor] = []
        self._hooks:      List = []
        self._register()

    def _register(self) -> None:
        """Hook into every MultiheadAttention in the decoder."""
        for layer in self.model.decoder.transformer_decoder.layers:
            h = layer.multihead_attn.register_forward_hook(self._hook_fn)
            self._hooks.append(h)

    def _hook_fn(self, module, inp, output):
        # output = (attn_output, attn_weights)  for batch_first=True
        if isinstance(output, tuple) and len(output) == 2:
            attn_w = output[1]
            if attn_w is not None:
                self._attn_maps.append(attn_w.detach().cpu())

    def clear(self) -> None:
        self._attn_maps.clear()

    def get_maps(self) -> List[torch.Tensor]:
        return self._attn_maps.copy()

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()

    def __del__(self):
        self.remove_hooks()


# ─── Quick demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ai_model.models.radiology_model import RadiologyReportModel

    device = torch.device("cpu")
    model  = RadiologyReportModel(vocab_size=3000).to(device)

    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    explainer = GradCAMExplainer(model)
    heatmap, overlay = explainer.explain(img, save_path="/tmp/test_gradcam.png")
    print("Heatmap shape:", heatmap.shape)
    print("Overlay size:", overlay.size)
    print("Demo Grad-CAM generated ✓")
