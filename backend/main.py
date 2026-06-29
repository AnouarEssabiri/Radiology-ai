"""
backend/main.py
================
FastAPI application entry-point.

Endpoints:
  POST /predict   – Upload X-ray → receive generated report + Grad-CAM
  POST /train     – Trigger a training run (async background task)
  GET  /evaluate  – Evaluate model on test set
  GET  /health    – Liveness probe

Run:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

# Updated 2026-06-29: Added full model deployment
import io
import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# ── project path ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.schemas import PredictResponse, EvaluateResponse, TrainResponse
from backend.services.inference_service import get_engine
from backend.utils.image_utils import validate_image, convert_to_pil, image_to_base64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("API")

# ── Config ────────────────────────────────────────────────────────────────────
CFG_PATH = os.environ.get("CONFIG_PATH", "config/config.yaml")
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

UPLOAD_DIR = Path(CFG["api"]["upload_dir"])
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

GRADCAM_DIR = Path(CFG["explainability"]["output_dir"])
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title        = "RadiologyAI API",
    description  = "Automatic Radiology Report Generation from Chest X-Rays",
    version      = "1.0.0",
    docs_url     = "/docs",
    redoc_url    = "/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
cors_origins = CFG["api"]["cors_origins"]
allow_credentials = True

# If wildcard is in origins, handle it appropriately to avoid FastAPI errors
if "*" in cors_origins or any(o.strip().lower() == "*" for o in cors_origins if isinstance(o, str)):
    allow_origins = ["*"]
    allow_credentials = False
else:
    # Ensure null (for file://) and common ports are allowed to minimize connection issues
    allow_origins = list(cors_origins)
    for origin in [
        "null",
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5500", "http://127.0.0.1:5500",
        "http://localhost:8000", "http://127.0.0.1:8000",
        "http://localhost:8080", "http://127.0.0.1:8080",
    ]:
        if origin not in allow_origins:
            allow_origins.append(origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = allow_origins,
    allow_credentials = allow_credentials,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Static files (Grad-CAM overlays) ──────────────────────────────────────────
# Mount gradcam dir only when it exists (created on first inference)
import os as _os
if _os.path.isdir(str(GRADCAM_DIR)):
    app.mount("/static/gradcam", StaticFiles(directory=str(GRADCAM_DIR)), name="gradcam")
else:
    @app.on_event("startup")
    async def mount_gradcam():
        GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
        if not any(r.path == "/static/gradcam" for r in app.routes):
            from starlette.routing import Mount
            app.mount("/static/gradcam", StaticFiles(directory=str(GRADCAM_DIR)), name="gradcam")


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("RadiologyAI API starting up …")
    # Pre-load the model so the first request is fast
    try:
        get_engine(CFG_PATH)
        logger.info("Model loaded ✓")
    except Exception as exc:
        logger.warning("Model pre-load failed (will retry on first request): %s", exc)


# ─── Frontend ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Frontend"])
async def serve_index():
    """Serve the frontend index.html"""
    frontend_dir = Path(__file__).parent.parent / "frontend"
    return FileResponse(frontend_dir / "index.html")

# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

@app.get("/healthz", tags=["System"])
async def healthz_check():
    """Health check endpoint for Render monitoring"""
    return {"status": "ok", "version": "1.0.0"}


# ─── Predict ──────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(
    file:       UploadFile = File(..., description="Chest X-ray image (JPG/PNG/DCM)"),
    beam_size:  int = 5,
    gradcam:    bool = True,
):
    """
    Upload a chest X-ray and receive an auto-generated radiology report.

    - **file**: image file (JPEG, PNG, or DICOM)
    - **beam_size**: beam search width (3–10)
    - **gradcam**: whether to generate a Grad-CAM overlay
    """
    # ── Validate ───────────────────────────────────────────────────────────
    max_bytes = CFG["api"]["max_file_size_mb"] * 1024 * 1024
    content   = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(413, f"File too large (max {CFG['api']['max_file_size_mb']} MB)")

    ext = Path(file.filename or "img.png").suffix.lower()
    if ext not in CFG["api"]["allowed_extensions"]:
        raise HTTPException(415, f"Unsupported file type: {ext}")

    # ── Convert to PIL ─────────────────────────────────────────────────────
    try:
        pil_image = convert_to_pil(content, ext)
        validate_image(pil_image)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    # ── Save upload ────────────────────────────────────────────────────────
    uid      = uuid.uuid4().hex
    img_path = UPLOAD_DIR / f"{uid}{ext}"
    with open(img_path, "wb") as f_out:
        f_out.write(content)

    # ── Inference ──────────────────────────────────────────────────────────
    try:
        engine = get_engine(CFG_PATH)
        result = engine.predict(pil_image, beam_size=beam_size, generate_gradcam=gradcam)
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(500, f"Inference failed: {exc}")

    # ── Save Grad-CAM ──────────────────────────────────────────────────────
    gradcam_url = None
    if result["gradcam_pil"] is not None:
        gc_path = GRADCAM_DIR / f"{uid}_gradcam.png"
        result["gradcam_pil"].save(gc_path)
        gradcam_url = f"/static/gradcam/{uid}_gradcam.png"

    # ── Encode original image as base64 for frontend ──────────────────────
    orig_b64 = image_to_base64(pil_image)

    return PredictResponse(
        uid            = uid,
        report         = result["report"],
        confidence     = result["confidence"],
        gradcam_url    = gradcam_url,
        original_b64   = orig_b64,
        latency_ms     = result["latency_ms"],
    )


# ─── Evaluate ─────────────────────────────────────────────────────────────────
@app.get("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
async def evaluate():
    """
    Run the model against the held-out test set and return evaluation metrics.
    This may take several minutes on large datasets.
    """
    try:
        engine  = get_engine(CFG_PATH)
        metrics = engine.evaluate_on_test_set()
        return EvaluateResponse(metrics=metrics)
    except Exception as exc:
        logger.exception("Evaluation error")
        raise HTTPException(500, str(exc))


# ─── Train (background) ───────────────────────────────────────────────────────
_training_status: dict = {"running": False, "message": "idle"}


@app.post("/train", response_model=TrainResponse, tags=["Training"])
async def trigger_training(background_tasks: BackgroundTasks):
    """
    Kick off a background training run.
    Returns immediately; poll /train/status for progress.
    """
    if _training_status["running"]:
        raise HTTPException(409, "A training run is already in progress.")

    def _run_training():
        _training_status["running"] = True
        _training_status["message"] = "training started"
        try:
            import subprocess
            subprocess.run(
                ["python", "ai_model/training/trainer.py", "--config", CFG_PATH],
                check=True,
            )
            _training_status["message"] = "training complete"
        except Exception as exc:
            _training_status["message"] = f"training failed: {exc}"
        finally:
            _training_status["running"] = False

    background_tasks.add_task(_run_training)
    return TrainResponse(status="started", message="Training launched in background.")


@app.get("/train/status", tags=["Training"])
async def training_status():
    return _training_status


# ─── Gradcam image download ───────────────────────────────────────────────────
@app.get("/gradcam/{filename}", tags=["Inference"])
async def get_gradcam_image(filename: str):
    path = GRADCAM_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Grad-CAM image not found")
    return FileResponse(path, media_type="image/png")
