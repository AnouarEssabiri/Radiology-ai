"""
backend/services/inference_service.py
========================================
Singleton accessor for the RadiologyInferenceEngine.
Called from FastAPI route handlers.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_engine = None


def get_engine(config_path: str = "config/config.yaml"):
    """
    Return the global inference engine, loading it on first call.
    Thread-safe for single-worker deployments.
    """
    global _engine
    if _engine is None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from ai_model.inference import RadiologyInferenceEngine
        _engine = RadiologyInferenceEngine.from_config(config_path)
        logger.info("Inference engine initialised.")
    return _engine
