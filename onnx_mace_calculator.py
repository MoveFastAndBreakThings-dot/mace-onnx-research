"""
ASE Calculator wrapping mace_dynamic.onnx.

Drop-in replacement for MACECalculator (energy only) — runs ONNX instead of PyTorch.
Forces not available: ONNX has no autograd.
"""

import numpy as np
import onnxruntime as ort
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list

N_ELEMENTS = 89   # MACE-MP-0 covers Z=1..89
R_MAX      = 6.0  # Å — MACE-MP-0 small cutoff


class ONNXMACECalculator(Calculator):
    implemented_properties = ["energy"]

    def __init__(self, model_path="mace_dynamic.onnx", r_max=R_MAX, **kwargs):
        super().__init__(**kwargs)
        self.sess  = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.r_max = r_max

    # ── Build ONNX feed from ASE Atoms ────────────────────────────────────────

    def _build_feed(self, atoms):
        n = len(atoms)
        z = atoms.get_atomic_numbers()

        # One-hot node features  [n_atoms, 89]
        node_attrs = np.zeros((n, N_ELEMENTS), dtype=np.float32)
        for i, zi in enumerate(z):
            node_attrs[i, zi - 1] = 1.0

        positions = atoms.get_positions().astype(np.float32)

        # Neighbour list — returns sender i, receiver j, integer cell shifts S
        src, dst, S = neighbor_list("ijS", atoms, self.r_max)
        cell = atoms.get_cell().array.astype(np.float32)  # [3, 3]

        # Actual shift vectors in Å
        shifts     = (S @ cell).astype(np.float32)        # [n_edges, 3]
        edge_index = np.array([src, dst], dtype=np.int64) # [2, n_edges]
        batch      = np.zeros(n, dtype=np.int64)

        return {
            "node_attrs": node_attrs,
            "positions":  positions,
            "shifts":     shifts,
            "edge_index": edge_index,
            "batch":      batch,
        }

    # ── ASE Calculator interface ──────────────────────────────────────────────

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        feed = self._build_feed(self.atoms)
        result = self.sess.run(None, feed)
        self.results["energy"] = float(np.array(result[0]).flat[0])
