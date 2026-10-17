#!/usr/bin/env bash
set -e

echo "=== RespiraTwin Workflow Runner ==="

echo "1. Generating synthetic patient EHR data..."
python src/generate_data.py --num_patients 300 --seed 42 --output_dir data

echo "2. Running unit tests and data validation suite..."
pytest -v

echo "3. Training model (stub for later stages)..."
# python src/train_model.py

echo "4. Running dashboard app..."
streamlit run app.py
