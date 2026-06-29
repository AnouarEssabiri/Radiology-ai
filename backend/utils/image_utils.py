"""
backend/utils/image_utils.py
==============================
Image validation, conversion, and base64 helpers.
Supports JPEG, PNG, and DICOM formats.
"""

import base64
import io
import logging
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_MODES = {"RGB", "L", "RGBA"}
MIN_SIZE = 64    # pixels
MAX_SIZE = 4096  # pixels


def convert_to_pil(content: bytes, ext: str) -> Image.Image:
    """
    Convert raw bytes to a PIL Image.
    Handles JPEG, PNG, and DICOM (.dcm) formats.
    """
    ext = ext.lower()

    if ext == ".dcm":
        return _dicom_to_pil(content)

    try:
        return Image.open(io.BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc


def _dicom_to_pil(content: bytes) -> Image.Image:
    """Convert DICOM bytes → PIL Image using pydicom."""
    try:
        import pydicom
        import numpy as np
        dcm  = pydicom.dcmread(io.BytesIO(content))
        arr  = dcm.pixel_array.astype(np.float32)

        # Normalise to 0-255
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        arr = (arr * 255).astype(np.uint8)

        img = Image.fromarray(arr)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except ImportError:
        raise ValueError("pydicom required for DICOM support: pip install pydicom")
    except Exception as exc:
        raise ValueError(f"DICOM conversion error: {exc}") from exc


def validate_image(image: Image.Image) -> None:
    """
    Raise ValueError if the image doesn't meet requirements.
    """
    w, h = image.size
    if w < MIN_SIZE or h < MIN_SIZE:
        raise ValueError(f"Image too small: {w}×{h}px (min {MIN_SIZE}×{MIN_SIZE})")
    if w > MAX_SIZE or h > MAX_SIZE:
        raise ValueError(f"Image too large: {w}×{h}px (max {MAX_SIZE}×{MAX_SIZE})")
    if image.mode not in SUPPORTED_MODES:
        # Attempt conversion rather than rejecting
        logger.debug("Converting %s → RGB", image.mode)


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL Image as a base64 data URI string."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format=fmt)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def base64_to_pil(data_uri: str) -> Image.Image:
    """Decode a base64 data URI back to a PIL Image."""
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    img_bytes = base64.b64decode(data_uri)
    return Image.open(io.BytesIO(img_bytes))
