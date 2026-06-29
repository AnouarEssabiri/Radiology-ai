@echo off
REM ================================================================
REM  RadiologyAI Windows All-In-One Fix and Launch Script
REM  Run from the project root:
REM    cd C:\Users\victus\Documents\radiology-ai-project\radiology-ai
REM    windows_fix.bat
REM ================================================================
setlocal enabledelayedexpansion

echo.
echo ================================================================
echo   RadiologyAI  Windows Setup and Fix
echo ================================================================

echo.
echo [1/6] Fixing Python Scripts PATH...
for /f "tokens=*" %%i in ('python -c "import sysconfig; print(sysconfig.get_path(\"scripts\"))"') do set PYSC=%%i
set PATH=%PYSC%;%PATH%
echo     Added: %PYSC%

echo.
echo [2/6] Installing required packages...
python -m pip install --quiet uvicorn[standard] fastapi python-multipart pydantic PyYAML tqdm numpy scipy pillow rouge-score nltk

echo.
echo [3/6] NLTK data...
python -c "import nltk; [nltk.download(p, quiet=True) for p in ['wordnet','omw-1.4','punkt','punkt_tab']]" 2>nul
echo     Done.

echo.
echo [4/6] Generating synthetic dataset...
if not exist "datasets\processed\annotations.json" (
    python datasets\scripts\generate_synthetic_dataset.py --n 500
) else (
    echo     Already exists, skipping.
)

echo.
echo [5/6] Quick training test...
if not exist "ai_model\checkpoints\best_model.pth" (
    python quick_train.py
) else (
    echo     Checkpoint exists, skipping.
)

echo.
echo [6/6] Starting backend API at http://localhost:8000 ...
echo       Open frontend\index.html in your browser after it starts.
echo       Press Ctrl+C to stop.
echo.
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

endlocal
