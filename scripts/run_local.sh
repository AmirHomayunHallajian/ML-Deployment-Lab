#!/usr/bin/env bash
set -euo pipefail

python scripts/generate_synthetic_data.py
python scripts/train_model.py
uvicorn app.main:app --reload
