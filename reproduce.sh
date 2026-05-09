#!/bin/bash
# =============================================================================
# reproduce.sh: Full reproduction from raw MIMIC-IV to final results.
#
# Requirements:
#   - MIMIC-IV CSV files in data/raw/ (from PhysioNet)
#   - Python environment with requirements.txt installed
#   - NVIDIA GPU (recommended; falls back to CPU)
#
# Usage:
#   bash reproduce.sh
#
# Expected runtime: ~8-16 hours on a single A100 GPU
# =============================================================================
set -e

echo "=============================="
echo "IMST-Mamba Reproduction Script"
echo "=============================="

# 0. Install dependencies
echo "[0/5] Installing dependencies..."
pip install -r requirements.txt --quiet

# 1. Data pipeline
echo "[1/5] Running data pipeline..."
python scripts/run_pipeline.py --stage data

# 2. Train all models (3 seeds each)
echo "[2/5] Training all models (9 models × 3 seeds)..."
for model in imst_mamba grud lstm transformer retain; do
    echo "  Training $model..."
    python scripts/run_pipeline.py --stage train --model $model --seeds 42 123 456
done

# 3. Evaluate
echo "[3/5] Running evaluation suite..."
python scripts/run_pipeline.py --stage evaluate

# 4. Generate figures (if notebooks are available)
echo "[4/5] Done! Results saved to results/"

echo "=============================="
echo "Reproduction complete."
echo "See results/evaluation_results.json for main metrics."
echo "=============================="
