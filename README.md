<div align="center">

# ⚛️ MACE-MP-0 · Browser-Native ML Force Field

**Run a state-of-the-art machine learning force field entirely in your browser.**
No Python. No PyTorch. No installation. Just open a tab.

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://python.org)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime%20Web-orange?style=for-the-badge&logo=onnx)](https://onnxruntime.ai)
[![Marimo](https://img.shields.io/badge/Marimo-Notebook-purple?style=for-the-badge)](https://marimo.io)
[![ASE](https://img.shields.io/badge/ASE-Calculator-green?style=for-the-badge)](https://wiki.fysik.dtu.dk/ase/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

```
  H₂O  →  MACE-MP-0 (ONNX)  →  −14.047934 eV   ✓ exact match
  CH₄  →  MACE-MP-0 (ONNX)  →  −23.766247 eV   ✓ exact match
  C₆H₆ →  MACE-MP-0 (ONNX)  →  −76.060577 eV   ✓ exact match
        all running in your browser tab
```

</div>

---

## 🧠 What Is This?

[MACE-MP-0](https://arxiv.org/abs/2401.00096) is a universal ML force field trained on the Materials Project — it predicts molecular energies with near-quantum accuracy across the entire periodic table (3.8M parameters, covering 89 elements).

The problem: MACE runs on **PyTorch**, which is ~500MB and impossible to run in a browser.

This project converts MACE to **ONNX** format and deploys it in a **Marimo browser notebook** using **ONNX Runtime Web (WebAssembly)** — so anyone can run ML-quality molecular energy predictions by opening a link.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Browser Tab                          │
│                                                         │
│   ┌─────────────────────────────────────────────────┐   │
│   │          Marimo Notebook (Python / WASM)        │   │
│   │                                                 │   │
│   │   atoms = molecule("H2O")                       │   │
│   │   atoms.calc = ONNXMACECalculator()             │   │
│   │   energy = atoms.get_potential_energy()  ──┐    │   │
│   │                                            │    │   │
│   │        ┌───────────────────────────────────┘    │   │
│   │        ▼         anywidget bridge               │   │
│   │   ┌──────────────────────────────────────┐      │   │
│   │   │   onnxruntime-web (JavaScript/WASM)  │      │   │
│   │   │   mace_dynamic.onnx (16 MB)          │      │   │
│   │   │   → energy: −14.047934 eV            │      │   │
│   │   └──────────────────────────────────────┘      │   │
│   └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

   No server. No Python runtime. No GPU. Just a browser.
```

---

## 📊 Results

Energies match PyTorch MACE to **float32 precision** across all molecules:

| Molecule | Atoms | Graph Edges | MACE (eV) | ONNX (eV) | \|Δ\| (eV) |
|----------|:-----:|:-----------:|:---------:|:---------:|:----------:|
| H₂O | 3 | 6 | −14.047934 | −14.047934 | 0.0e+00 |
| CO₂ | 3 | 6 | −22.829908 | −22.829908 | 0.0e+00 |
| NH₃ | 4 | 12 | −19.506739 | −19.506737 | 1.9e-06 |
| CH₄ | 5 | 20 | −23.766247 | −23.766247 | 0.0e+00 |
| H₂O₂ | 4 | 12 | −17.942633 | −17.942635 | 1.9e-06 |
| **C₆H₆** | **12** | **132** | **−76.060577** | **−76.060577** | **3.9e-07** |

**Speed (Python CPU, avg 10 runs):**

| Molecule | MACE/PyTorch | ONNX Runtime |
|----------|:------------:|:------------:|
| H₂O | 23.7 ms | 20.9 ms |
| C₆H₆ | 28.6 ms | 105.1 ms |

> Note: ONNX is slower for larger molecules due to `scatter_add` workaround ops (GatherND/ScatterElements). A native scatter kernel in ONNX would fix this — documented as a finding.

---

## 🚀 Quick Start

### Browser Demo (HTML)
```bash
git clone https://github.com/MoveFastAndBreakThings-dot/mace-onnx-research
cd mace-onnx-research
python serve.py
# Open http://localhost:8080/demo.html
```

### Marimo Notebook
```bash
pip install marimo anywidget
marimo edit marimo_demo.py
# Open http://localhost:2718
```

### ASE Calculator (Python)
```python
from onnx_mace_calculator import ONNXMACECalculator
from ase.build import molecule

atoms = molecule("C6H6")
atoms.center(vacuum=10.0)
atoms.calc = ONNXMACECalculator("mace_dynamic.onnx")

energy = atoms.get_potential_energy()
print(f"Energy: {energy:.6f} eV")
# Energy: -76.060577 eV
```

---

## 📁 File Structure

```
mace-onnx-research/
│
├── mace_dynamic.onnx          # ONNX model with dynamic axes (16 MB)
├── onnx_mace_calculator.py    # ASE Calculator wrapper
├── marimo_demo.py             # Interactive browser notebook
├── demo.html                  # Standalone browser demo
│
├── export_mace.py             # MACE → ONNX conversion script
├── molecules.json             # Pre-computed inputs + reference energies
├── inputs_*.json              # Per-molecule graph tensors (browser-ready)
│
└── visualize.py               # PyTorch vs ONNX comparison plot
```

---

## 🔬 Key Technical Findings

**1. Dynamic graph export**
`torch.jit.trace` bakes in graph topology. Re-exported with `dynamic_axes` on all variable dimensions (`n_atoms`, `n_edges`) — same 16MB model now handles any molecule.

**2. scatter_add workaround**
MACE uses `scatter_add` for message passing. PyTorch's version breaks ONNX export due to dynamic shapes. Fixed by registering a custom `GatherND`+`ScatterElements` symbolic. Works correctly but adds overhead for large graphs.

**3. Forces unavailable**
MACE computes forces via autograd (backpropagation through PyTorch). ONNX strips the computation graph — only the forward pass survives. Energy-only inference is the practical limit of this export approach.

**4. anywidget bridge pattern**
Pyodide (browser Python) has no `onnxruntime`. Solution: anywidget syncs data between Python and JavaScript via traitlets. JS runs `onnxruntime-web`, Python orchestrates and displays. The model is passed as base64 to avoid file-serving complexity.

---

## 🛠️ Reproducing the Export

```bash
# Set up environment
bash setup_env.sh
source venv/bin/activate

# Convert MACE → ONNX (requires mace-torch, torch)
python export_mace.py

# Test accuracy across molecules
python test_onnx.py

# Generate comparison plot
python visualize.py
```

---

## 📖 References

- [MACE-MP-0 paper](https://arxiv.org/abs/2401.00096) — Batatia et al., 2023
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) — browser ML inference
- [Marimo](https://marimo.io) — reactive Python notebooks
- [Kermode marimo-jax-onnx-demo](https://github.com/jameskermode/marimo-jax-onnx-demo) — anywidget pattern reference
- [ASE](https://wiki.fysik.dtu.dk/ase/) — Atomic Simulation Environment

---

<div align="center">

**University of Alberta · MSc Research · 2025–2026**

*Supervisor: Tian Tian*

</div>
