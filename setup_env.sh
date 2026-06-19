#!/bin/bash
# Creates a clean Python venv and installs everything needed
# Run once. Takes ~5 minutes (downloads MACE + deps).

set -e

echo "=== Creating virtual environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Skipping PyTorch (using system torch via venv access) ==="
# System already has torch 2.12 — inherit it via --system-site-packages below
# (venv was created without it, so we install torch from PyPI default index)
pip install torch --index-url https://pypi.org/simple/

echo "=== Installing MACE ==="
pip install mace-torch

echo "=== Installing ONNX + runtime ==="
pip install onnx onnxruntime onnxscript

echo "=== Installing ASE (needed by MACE) ==="
pip install ase

echo ""
echo "=== Done! ==="
echo "Next: source venv/bin/activate && python export_mace.py"
