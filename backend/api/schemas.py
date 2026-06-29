"""
backend/api/schemas.py
========================
Pydantic models for request validation and response serialisation.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    uid:          str           = Field(..., description="Unique inference ID")
    report:       str           = Field(..., description="Auto-generated radiology report")
    confidence:   float         = Field(..., ge=0.0, le=1.0,
                                       description="Model confidence score [0-1]")
    gradcam_url:  Optional[str] = Field(None, description="URL to Grad-CAM overlay image")
    original_b64: Optional[str] = Field(None, description="Original image (base64 PNG)")
    latency_ms:   float         = Field(..., description="Inference latency in milliseconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "uid":        "a3f2e1b4c5d6",
                "report":     "The lungs are clear. No pleural effusion or pneumothorax is seen. "
                              "Cardiomediastinal silhouette is within normal limits.",
                "confidence": 0.87,
                "gradcam_url": "/static/gradcam/a3f2e1b4c5d6_gradcam.png",
                "latency_ms": 342.5,
            }
        }
    }


class EvaluateResponse(BaseModel):
    metrics: Dict[str, float] = Field(
        ...,
        description="Evaluation metrics (BLEU-1/2/3/4, ROUGE, METEOR, accuracy)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "metrics": {
                    "bleu_1": 0.42,
                    "bleu_4": 0.18,
                    "rouge1": 0.46,
                    "rougeL": 0.38,
                    "meteor": 0.31,
                }
            }
        }
    }


class TrainResponse(BaseModel):
    status:  str = Field(..., description="Job status: started | running | failed")
    message: str = Field(..., description="Human-readable status message")


class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str = "1.0.0"
