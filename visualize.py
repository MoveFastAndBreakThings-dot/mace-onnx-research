import matplotlib
matplotlib.use("Agg")  # no display needed — saves to file
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mace.calculators import mace_mp
from ase.build import molecule
import onnxruntime as ort

# ── Load model and molecule ───────────────────────────────────────────────────
calc = mace_mp(model="small", device="cpu", default_dtype="float32")
model = calc.models[0]
model.eval()

atoms = molecule("H2O")
atoms.center(vacuum=10.0)

# ── Capture inputs ────────────────────────────────────────────────────────────
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

# ── MACE: energy + forces ─────────────────────────────────────────────────────
result = model(input_dict, compute_force=True)
mace_energy = result["energy"].item()
mace_forces = result["forces"].detach().numpy()

# ── ONNX: energy only ─────────────────────────────────────────────────────────
sess = ort.InferenceSession("mace_small.onnx")
all_tensors = {
    "node_attrs":  input_dict["node_attrs"],
    "positions":   input_dict["positions"].detach(),
    "shifts":      input_dict["shifts"],
    "unit_shifts": input_dict["unit_shifts"],
    "edge_index":  input_dict["edge_index"],
    "batch":       input_dict["batch"],
    "ptr":         input_dict["ptr"],
    "cell":        input_dict["cell"],
    "pbc":         input_dict["pbc"],
}
feed = {i.name: all_tensors[i.name].detach().numpy() for i in sess.get_inputs()}
onnx_energy = float(np.array(sess.run(None, feed)[0]).flat[0])

# ── Atom positions and types ──────────────────────────────────────────────────
pos = input_dict["positions"].detach().numpy()  # (3, 3)
atom_symbols = ["O", "H", "H"]
atom_colors  = ["red", "lightblue", "lightblue"]
atom_sizes   = [800, 400, 400]

# Use XY plane for 2D view
x, y = pos[:, 0], pos[:, 1]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#1a1a2e")

def style_ax(ax, title):
    ax.set_facecolor("#16213e")
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("X position (Å)", color="#aaaaaa")
    ax.set_ylabel("Y position (Å)", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_color("#444466")

# ── LEFT: MACE (energy + forces) ─────────────────────────────────────────────
style_ax(ax1, "MACE (PyTorch)  —  Energy + Forces")

# Draw bonds
for i in range(1, 3):
    ax1.plot([x[0], x[i]], [y[0], y[i]], color="#888888", lw=2, zorder=1)

# Draw atoms
for i in range(3):
    ax1.scatter(x[i], y[i], s=atom_sizes[i], c=atom_colors[i],
                edgecolors="white", linewidths=1.5, zorder=3)
    ax1.text(x[i], y[i], atom_symbols[i], ha="center", va="center",
             fontsize=12, fontweight="bold", color="black", zorder=4)

# Draw force arrows
scale = 3.0
for i in range(3):
    fx, fy = mace_forces[i, 0], mace_forces[i, 1]
    mag = np.sqrt(fx**2 + fy**2)
    ax1.annotate("", xy=(x[i] + fx*scale, y[i] + fy*scale),
                 xytext=(x[i], y[i]),
                 arrowprops=dict(arrowstyle="->", color="#00ff88",
                                 lw=2.0, mutation_scale=15),
                 zorder=5)
    ax1.text(x[i] + fx*scale*1.15, y[i] + fy*scale*1.15,
             f"{mag:.3f} eV/Å", color="#00ff88", fontsize=8, ha="center")

# Energy label
ax1.text(0.5, 0.04, f"Energy = {mace_energy:.4f} eV",
         transform=ax1.transAxes, ha="center", color="#ffdd44",
         fontsize=12, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2a4a", edgecolor="#ffdd44"))

# Legend
green_patch = mpatches.Patch(color="#00ff88", label="Force vectors (eV/Å)")
ax1.legend(handles=[green_patch], loc="upper right",
           facecolor="#2a2a4a", edgecolor="#555577", labelcolor="white")

ax1.set_aspect("equal")
ax1.margins(0.3)

# ── RIGHT: ONNX (energy only) ─────────────────────────────────────────────────
style_ax(ax2, "ONNX (Browser-ready)  —  Energy only")

# Draw bonds
for i in range(1, 3):
    ax2.plot([x[0], x[i]], [y[0], y[i]], color="#888888", lw=2, zorder=1)

# Draw atoms
for i in range(3):
    ax2.scatter(x[i], y[i], s=atom_sizes[i], c=atom_colors[i],
                edgecolors="white", linewidths=1.5, zorder=3)
    ax2.text(x[i], y[i], atom_symbols[i], ha="center", va="center",
             fontsize=12, fontweight="bold", color="black", zorder=4)

# No force arrows — show "missing" label instead
ax2.text(0.5, 0.75, "Forces: NOT available\n(ONNX has no autograd)",
         transform=ax2.transAxes, ha="center", color="#ff6666",
         fontsize=10, style="italic",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#3a1a1a", edgecolor="#ff6666"))

# Energy label
diff = abs(onnx_energy - mace_energy)
ax2.text(0.5, 0.04, f"Energy = {onnx_energy:.4f} eV   (diff = {diff:.0e} eV)",
         transform=ax2.transAxes, ha="center", color="#ffdd44",
         fontsize=12, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2a4a", edgecolor="#ffdd44"))

ax2.set_aspect("equal")
ax2.margins(0.3)

# ── Title + caption ───────────────────────────────────────────────────────────
fig.suptitle("MACE-MP-0 on H₂O:  PyTorch vs ONNX export",
             color="white", fontsize=16, fontweight="bold", y=1.01)

fig.text(0.5, -0.04,
         "Green arrows = forces on each atom (direction + magnitude).  "
         "ONNX gives same energy but cannot compute forces yet.",
         ha="center", color="#aaaaaa", fontsize=10)

plt.tight_layout()
plt.savefig("comparison.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved: comparison.png")
