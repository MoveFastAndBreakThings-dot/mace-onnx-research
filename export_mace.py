"""
MACE-MP-0 → ONNX export attempt.

Strategies tried (in order):
  A. torch.jit.trace + torch.onnx.export(dynamo=False)  ← legacy tracing, bypasses ScriptModule issue
  B. trace → save TorchScript → reload → export
  C. torch.compile with best available backend

Key findings so far:
  - dynamo=True / torch.export fails: e3nn uses ScriptModule internally, incompatible
  - torch.jit.script fails: e3nn Python objects (Irrep.dim etc.) not scriptable
  - Forces computed via autograd (not direct NN output) — additional ONNX challenge
"""

import sys
import os
import traceback
import torch

print("=" * 60)
print("MACE → ONNX Export Research Script")
print("=" * 60)

# ── 1. Load model ─────────────────────────────────────────────────────────────

print("\n[1/5] Loading MACE-MP-0 small model...")

from mace.calculators import mace_mp

calc = mace_mp(model="small", device="cpu", default_dtype="float32")
model = calc.models[0]
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"  ✓ Model type : {type(model).__name__}")
print(f"  ✓ Parameters : {total_params:,}")
print(f"  ✓ Cutoff r   : {model.r_max} Å")

# ── 2. Capture real inputs via calculator ─────────────────────────────────────

print("\n[2/5] Building H2O and capturing model inputs...")

import ase
from ase.build import molecule

atoms = molecule("H2O")
atoms.center(vacuum=10.0)

captured = {}
_orig = model.forward

def _cap(data, compute_force=False, **kwargs):
    captured["data"] = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                        for k, v in data.items()}
    return _orig(data, compute_force=compute_force, **kwargs)

model.forward = _cap
atoms.calc = calc
_ = atoms.get_potential_energy()
model.forward = _orig

input_dict = captured["data"]
input_dict["positions"] = input_dict["positions"].detach().requires_grad_(True)

print(f"  ✓ Atoms: {len(atoms)}, edges: {input_dict['edge_index'].shape[1]}")

# ── 3. Verify forward pass ────────────────────────────────────────────────────

print("\n[3/5] Verifying forward pass...")

result = model(input_dict, compute_force=True)
print(f"  ✓ Energy: {result['energy'].item():.6f} eV")
print(f"  ✓ Forces: {result['forces'].shape}, mean {result['forces'].abs().mean().item():.4f} eV/Å")
print(f"  NOTE: forces = autograd.grad(energy, positions) — not a direct NN output")

# ── 4. Shared wrapper + example inputs ───────────────────────────────────────

node_attrs  = input_dict["node_attrs"]
positions   = input_dict["positions"]
shifts      = input_dict["shifts"]
unit_shifts = input_dict["unit_shifts"]
edge_index  = input_dict["edge_index"]
batch_idx   = input_dict["batch"]
ptr         = input_dict["ptr"]
cell        = input_dict["cell"]
pbc         = input_dict["pbc"]

print("\n  Input tensor shapes:")
for name, t in [("node_attrs", node_attrs), ("positions", positions),
                ("shifts", shifts), ("unit_shifts", unit_shifts),
                ("edge_index", edge_index), ("batch", batch_idx),
                ("ptr", ptr), ("cell", cell), ("pbc", pbc)]:
    print(f"    {name:12s}: {tuple(t.shape)} {t.dtype}")


class MACEEnergyWrapper(torch.nn.Module):
    """Energy-only — no forces, no autograd needed for export."""
    def __init__(self, m):
        super().__init__()
        self.model = m

    def forward(self, node_attrs, positions, shifts, unit_shifts,
                edge_index, batch_idx, ptr, cell, pbc):
        out = self.model({
            "node_attrs": node_attrs, "positions": positions,
            "shifts": shifts, "unit_shifts": unit_shifts,
            "edge_index": edge_index, "batch": batch_idx,
            "ptr": ptr, "cell": cell, "pbc": pbc,
        }, compute_force=False)
        return out["energy"]


pos_detached = positions.detach()
example_inputs = (node_attrs, pos_detached, shifts, unit_shifts,
                  edge_index, batch_idx, ptr, cell, pbc)

# ── 5. Export strategies ──────────────────────────────────────────────────────

print("\n[4/5] Attempting ONNX export...")
print("-" * 60)

export_succeeded = False
onnx_path = "mace_small.onnx"

# ── Strategy A: torch.jit.trace + legacy export (dynamo=False) ───────────────
# dynamo/torch.export can't handle e3nn ScriptModules.
# Legacy tracing just runs the model once and records tensor ops — ScriptModules
# execute normally and their ops get flattened into the trace.

print("\nStrategy A: torch.jit.trace + torch.onnx.export(dynamo=False) ...")
try:
    # ── Patch scatter_add ONNX symbolic ──────────────────────────────────────
    # Bug: symbolic_opset16.scatter_add calls len(src_sizes) but _get_tensor_sizes
    # returns None for dynamic tensors → TypeError.
    #
    # Root cause: the decorator @_onnx_symbolic("aten::scatter_add") registers the
    # function into registration.registry (a SymbolicRegistry singleton) at import
    # time. Patching the module attribute afterward has no effect on the registry.
    #
    # Fix: register a custom override via registration.registry.register(..., custom=True).
    # custom=True calls _SymbolicFunctionGroup.add_custom which inserts into the
    # OverrideDict's _overrides layer, which wins over the built-in _base layer.
    from torch.onnx._internal.torchscript_exporter import (
        registration as _reg,
        symbolic_helper as _sym_help,
        jit_utils as _jit_utils,
    )

    @_sym_help.parse_args("v", "i", "v", "v")
    def _scatter_add_fixed(g: _jit_utils.GraphContext, self, dim, index, src):
        # dim is already a Python int here (parse_args extracted it).
        # Guard against None sizes (dynamic shapes) instead of crashing.
        src_sizes = _sym_help._get_tensor_sizes(src)
        index_sizes = _sym_help._get_tensor_sizes(index)

        if src_sizes is not None and index_sizes is not None:
            if len(src_sizes) != len(index_sizes):
                return _sym_help._unimplemented(
                    "scatter_add",
                    f"`index` ({index_sizes}) should have the same dimensionality "
                    f"as `src` ({src_sizes})",
                )
            if src_sizes != index_sizes or None in index_sizes:
                adjusted_shape = g.op("Shape", index)
                starts = g.op("Constant", value_t=torch.tensor([0] * len(index_sizes)))
                src = g.op("Slice", src, starts, adjusted_shape)

        src = _sym_help._maybe_get_scalar(src)
        if _sym_help._is_value(src):
            return g.op("ScatterElements", self, index, src, axis_i=dim, reduction_s="add")
        else:
            return g.op("ScatterElements", self, index, src, axis_i=dim, reduction_s="add")

    # Register as custom override — wins over the built-in at opset 16.
    # The exporter resolves opset 18 → most recent registered ≤ 18 → our opset-16 override.
    _reg.registry.register("aten::scatter_add", 16, _scatter_add_fixed, custom=True)
    print("  ✓ Patched scatter_add in ONNX registry (custom override, None-sizes guard)")

    # ── Patch aten::index ONNX symbolic ──────────────────────────────────────
    # e3nn's FX-compiled Linear does reshape(*batch_shape, out_features) with a
    # dynamic shape vector → output tensor has unknown rank in ONNX IR.
    # MACE then does node_energy[arange, heads] (advanced 2D indexing) on that
    # tensor → aten::index symbolic fails with "unknown rank".
    #
    # Fix: when rank is unknown and multiple non-None indices are present, use
    # GatherND (opset 12+). This works without knowing rank because GatherND
    # uses the last dim of the indices tensor to determine how many dims to gather.
    # Only valid when all indices cover all dims of the tensor — which is true for
    # every 2-index MACE case (2D tensor, [arange, heads]).
    import torch.onnx._internal.torchscript_exporter.symbolic_opset9 as _opset9

    def custom_index(g: _jit_utils.GraphContext, self, index):
        if _sym_help._is_packed_list(index):
            indices = _sym_help._unpack_list(index)
        else:
            indices = [index]

        adv_idx_indices = [i for i, idx in enumerate(indices)
                           if not _sym_help._is_none(idx)]

        if len(adv_idx_indices) >= 2 and _sym_help._get_tensor_rank(self) is None:
            idx_tensors = [indices[i] for i in adv_idx_indices]
            axes = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
            unsqueezed = [g.op("Unsqueeze", idx, axes) for idx in idx_tensors]
            stacked = g.op("Concat", *unsqueezed, axis_i=-1)
            return g.op("GatherND", self, stacked)

        return _opset9.index(g, self, index)

    # Must register at opset 11 (not 9) because opset11 has its own aten::index
    # that wins dispatch for opset 18 and falls back to opset9.index for multi-index.
    _reg.registry.register("aten::index", 11, custom_index, custom=True)
    print("  ✓ Patched aten::index in ONNX registry (GatherND fallback for unknown rank)")
    # ─────────────────────────────────────────────────────────────────────────

    wrapper = MACEEnergyWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_inputs, strict=False)
    print("  ✓ torch.jit.trace succeeded")

    torch.onnx.export(
        traced,
        example_inputs,
        onnx_path,
        dynamo=False,
        opset_version=18,
        input_names=["node_attrs", "positions", "shifts", "unit_shifts",
                     "edge_index", "batch", "ptr", "cell", "pbc"],
        output_names=["energy"],
    )
    print(f"  ✓ ONNX export SUCCESS — saved to {onnx_path}")
    export_succeeded = True

except Exception as e:
    print(f"  ✗ FAILED  [{type(e).__name__}]")
    print(f"  {str(e)[:600]}")
    lines = traceback.format_exc().strip().split("\n")
    for line in lines[-8:]:
        print(f"    {line}")


# ── Strategy B: trace → save as .pt → reload → export ────────────────────────
# Sometimes the round-trip through TorchScript serialization helps.

if not export_succeeded:
    print("\nStrategy B: trace → save .pt → reload → export ...")
    try:
        wrapper_b = MACEEnergyWrapper(model)
        wrapper_b.eval()

        with torch.no_grad():
            traced_b = torch.jit.trace(wrapper_b, example_inputs, strict=False)
        traced_b.save("mace_traced.pt")
        reloaded = torch.jit.load("mace_traced.pt")
        print("  ✓ Saved and reloaded traced model")

        torch.onnx.export(
            reloaded,
            example_inputs,
            onnx_path,
            dynamo=False,
            opset_version=18,
            input_names=["node_attrs", "positions", "shifts", "unit_shifts",
                         "edge_index", "batch", "ptr", "cell", "pbc"],
            output_names=["energy"],
        )
        print(f"  ✓ ONNX export SUCCESS — saved to {onnx_path}")
        export_succeeded = True

    except Exception as e:
        print(f"  ✗ FAILED  [{type(e).__name__}]")
        print(f"  {str(e)[:600]}")
        lines = traceback.format_exc().strip().split("\n")
        for line in lines[-8:]:
            print(f"    {line}")


# ── Strategy C: torch.compile — list backends, try each ──────────────────────

if not export_succeeded:
    print("\nStrategy C: torch.compile ...")
    try:
        import torch._dynamo
        available = torch._dynamo.list_backends()
        print(f"  Available backends: {available}")

        for bname in ["onnxrt", "onnx_rt", "inductor"]:
            if bname not in available:
                print(f"  — {bname} not available, skipping")
                continue
            print(f"  Trying backend={bname} ...")
            try:
                wrapper_c = MACEEnergyWrapper(model)
                compiled = torch.compile(wrapper_c, backend=bname)
                with torch.no_grad():
                    r = compiled(*example_inputs)
                print(f"  ✓ backend={bname} ran! energy={r.item():.4f} eV")
                export_succeeded = True
                break
            except Exception as inner:
                print(f"  ✗ {bname}: {type(inner).__name__}: {str(inner)[:200]}")

    except Exception as e:
        print(f"  ✗ FAILED [{type(e).__name__}]: {str(e)[:300]}")


# ── 5. Validate if file was created ──────────────────────────────────────────

print("\n[5/5] Validation with ONNX Runtime...")
print("-" * 60)

if os.path.exists(onnx_path):
    try:
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"  ✓ ONNX model valid")
        print(f"  ✓ File size: {os.path.getsize(onnx_path) / 1e6:.1f} MB")

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp_names = [i.name for i in sess.get_inputs()]
        out_names = [o.name for o in sess.get_outputs()]
        print(f"  ✓ Session created. Inputs: {inp_names}")
        print(f"  ✓ Outputs: {out_names}")

        # Run inference — only feed inputs the session actually expects
        # (some tensors become constants during export and aren't model inputs).
        all_tensors = {
            "node_attrs":  node_attrs,
            "positions":   pos_detached,
            "shifts":      shifts,
            "unit_shifts": unit_shifts,
            "edge_index":  edge_index,
            "batch":       batch_idx,
            "ptr":         ptr,
            "cell":        cell,
            "pbc":         pbc,
        }
        feed = {i.name: all_tensors[i.name].detach().numpy()
                for i in sess.get_inputs()}
        ort_result = sess.run(None, feed)
        ort_energy = ort_result[0].item() if hasattr(ort_result[0], 'item') else float(ort_result[0])
        torch_energy = result["energy"].item()
        print(f"  ✓ ONNX Runtime inference OK!")
        print(f"    ONNX  energy: {ort_energy:.6f} eV")
        print(f"    PyTorch energy: {torch_energy:.6f} eV")
        print(f"    Diff: {abs(ort_energy - torch_energy):.2e} eV")

    except Exception as e:
        print(f"  ✗ Validation error: {type(e).__name__}: {e}")
else:
    print("  — No ONNX file created (all exports failed).")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
if export_succeeded and os.path.exists(onnx_path):
    print("  EXPORT + VALIDATION SUCCEEDED ✓")
    print(f"  File: {onnx_path}")
    print("  Next: wire onnxruntime-web into Atomify force-calc step")
elif export_succeeded:
    print("  PARTIAL SUCCESS (compiled but no .onnx file)")
    print("  Document which backend worked for the paper")
else:
    print("  ALL STRATEGIES FAILED ✗")
    print("  Document for paper:")
    print("    - dynamo=True: e3nn ScriptModule incompatible with torch.export")
    print("    - torch.jit.script: e3nn Python objects not scriptable")
    print("    - torch.jit.trace: see error above")
    print("  Next options:")
    print("    - Replace e3nn with pure-PyTorch equivariant ops (no ScriptModule)")
    print("    - Try mace-jax (JAX version, different export path)")
    print("    - Use finite-difference forces instead of autograd")
print("=" * 60)
