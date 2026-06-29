#!/usr/bin/env python3
"""
setup.py
=========
Quick environment setup script.

Run: python setup.py
"""

import os
import subprocess
import sys


def run(cmd: str, check: bool = True) -> int:
    print(f"\n▶ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"  ✗ Command failed (exit {result.returncode})")
        sys.exit(result.returncode)
    return result.returncode


def main():
    print("=" * 60)
    print("  RadiologyAI — Environment Setup")
    print("=" * 60)

    # Create directories
    dirs = [
        "datasets/raw",
        "datasets/processed/images",
        "datasets/processed/reports",
        "ai_model/checkpoints",
        "backend/uploads/gradcam",
        "logs/tensorboard",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  ✓ {d}")

    # Install Python dependencies
    run("pip install -r requirements.txt")

    # NLTK data
    run("python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')\"",
        check=False)

    print("\n" + "=" * 60)
    print("  Setup complete! Next steps:")
    print("=" * 60)
    print("  1. Prepare dataset:")
    print("     python datasets/scripts/prepare_dataset.py --download")
    print()
    print("  2. Train:")
    print("     python ai_model/training/trainer.py")
    print()
    print("  3. Start API:")
    print("     uvicorn backend.main:app --port 8000 --reload")
    print()
    print("  4. Open frontend:")
    print("     open frontend/index.html")
    print("=" * 60)


if __name__ == "__main__":
    main()
