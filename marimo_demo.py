import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import base64
    import pathlib
    import anywidget
    import traitlets
    from ase.build import molecule
    from ase.neighborlist import neighbor_list

    return (
        anywidget,
        base64,
        mo,
        molecule,
        neighbor_list,
        np,
        pathlib,
        traitlets,
    )


@app.cell
def _(mo):
    print(mo.__version__)
    return


@app.cell
def _(base64, pathlib):
    # Read ONNX model once and encode as base64 — passed to JS to avoid file-serving issues
    _model_path = pathlib.Path(__file__).parent / "mace_dynamic.onnx"
    with open(_model_path, "rb") as _f:
        MODEL_B64 = base64.b64encode(_f.read()).decode()
    print(f"Model loaded: {len(MODEL_B64) // 1024} KB (base64)")
    return (MODEL_B64,)


@app.cell
def _(anywidget, traitlets):
    ORT_VERSION = "1.24.3"

    class MACEWidget(anywidget.AnyWidget):
        _esm = f"""
    const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/ort.mjs";

    async function render({{ model, el }}) {{
    el.style.cssText = "font-family:monospace;font-size:13px;padding:6px 0;";
    el.textContent = "Loading MACE-MP-0 ONNX model...";

    // Load onnxruntime-web (ES module)
    const {{ default: ort }} = await import(ORT_URL);
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/`;

    // Decode base64 → ArrayBuffer → ORT session
    const b64 = model.get("model_b64");
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

    let session;
    try {{
        session = await ort.InferenceSession.create(bytes.buffer, {{
            executionProviders: ["wasm"],
        }});
        model.set("ready", true);
        model.save_changes();
        el.textContent = "MACE-MP-0 ready ✓  (WASM backend)";
        el.style.color = "#3fb950";
    }} catch (e) {{
        el.textContent = "Model load error: " + e.message;
        el.style.color = "#ff5555";
        return;
    }}

    async function runInference() {{
        const molData = model.get("molecule_data");
        if (!molData || Object.keys(molData).length === 0) return;

        el.textContent = "Running inference...";
        el.style.color = "#bc8cff";

        try {{
            const feeds = {{}};
            for (const [name, info] of Object.entries(molData)) {{
                const isInt = info.dtype === "int64";
                const raw = isInt
                    ? new BigInt64Array(info.data.map(v => BigInt(v)))
                    : new Float32Array(info.data);
                feeds[name] = new ort.Tensor(isInt ? "int64" : "float32", raw, info.shape);
            }}

            const t0 = performance.now();
            const results = await session.run(feeds);
            const ms = (performance.now() - t0).toFixed(1);

            const energy = Number(results["energy"].data[0]);
            model.set("energy", energy);
            model.set("inference_ms", ms);
            model.save_changes();

            el.textContent = `Done in ${{ms}} ms ✓`;
            el.style.color = "#3fb950";
        }} catch (e) {{
            el.textContent = "Inference error: " + e.message;
            el.style.color = "#ff5555";
        }}
    }}

    model.on("change:molecule_data", runInference);

    // Run immediately if molecule_data already set
    if (Object.keys(model.get("molecule_data")).length > 0) {{
        await runInference();
    }}
    }}

    export default {{ render }};
    """
        model_b64     = traitlets.Unicode("").tag(sync=True)
        molecule_data = traitlets.Dict({}).tag(sync=True)
        energy        = traitlets.Float(0.0).tag(sync=True)
        inference_ms  = traitlets.Unicode("—").tag(sync=True)
        ready         = traitlets.Bool(False).tag(sync=True)

    return (MACEWidget,)


@app.cell
def _():
    print('hi')
    return


@app.cell
def _(MACEWidget, MODEL_B64, mo):
    _w = MACEWidget(model_b64=MODEL_B64)
    mace_widget = mo.ui.anywidget(_w)
    return (mace_widget,)


@app.cell
def _():
    print('something liek that')
    return


@app.cell
def _(mo):
    mol_select = mo.ui.dropdown(
        options={
            "Water (H₂O)":              "H2O",
            "Carbon Dioxide (CO₂)":     "CO2",
            "Ammonia (NH₃)":            "NH3",
            "Methane (CH₄)":            "CH4",
            "Hydrogen Peroxide (H₂O₂)": "H2O2",
            "Benzene (C₆H₆)":           "C6H6",
        },
        value="Water (H₂O)",
        label="Molecule",
    )
    return (mol_select,)


@app.cell
def _(mace_widget, mol_select, molecule, neighbor_list, np):
    # Build MACE graph inputs from selected molecule and send to widget
    N_ELEMENTS = 89
    R_MAX = 6.0

    _mol_name = mol_select.value
    _atoms = molecule(_mol_name)
    _atoms.center(vacuum=10.0)

    _n = len(_atoms)
    _z = _atoms.get_atomic_numbers()

    _node_attrs = np.zeros((_n, N_ELEMENTS), dtype=np.float32)
    for _i, _zi in enumerate(_z):
        _node_attrs[_i, _zi - 1] = 1.0

    _positions  = _atoms.get_positions().astype(np.float32)
    _src, _dst, _S = neighbor_list("ijS", _atoms, R_MAX)
    _cell       = _atoms.get_cell().array.astype(np.float32)
    _shifts     = (_S @ _cell).astype(np.float32)
    _edge_index = np.array([_src, _dst], dtype=np.int64)
    _batch      = np.zeros(_n, dtype=np.int64)

    # Set on the underlying widget — triggers JS inference
    mace_widget.molecule_data = {
        "node_attrs": {"data": _node_attrs.flatten().tolist(), "shape": list(_node_attrs.shape), "dtype": "float32"},
        "positions":  {"data": _positions.flatten().tolist(),  "shape": list(_positions.shape),  "dtype": "float32"},
        "shifts":     {"data": _shifts.flatten().tolist(),     "shape": list(_shifts.shape),     "dtype": "float32"},
        "edge_index": {"data": _edge_index.flatten().tolist(), "shape": list(_edge_index.shape), "dtype": "int64"},
        "batch":      {"data": _batch.flatten().tolist(),      "shape": list(_batch.shape),      "dtype": "int64"},
    }
    return


@app.cell
def _(mace_widget, mo, mol_select):
    # Reference energies from Python MACE run (pre-computed)
    _REF = {
        "H2O":  -14.047934,
        "CO2":  -22.829908,
        "NH3":  -19.506739,
        "CH4":  -23.766247,
        "H2O2": -17.942633,
        "C6H6": -76.060577,
    }

    _mol  = mol_select.value
    _val  = mace_widget.value
    _e    = _val.get("energy", 0.0)
    _ms   = _val.get("inference_ms", "—")
    _ref  = _REF.get(_mol)
    _diff = abs(_e - _ref) if (_ref is not None and _e != 0.0) else None

    _energy_row = (
        mo.md(f"""
    | | |
    |---|---|
    | **Molecule** | {_mol} |
    | **ONNX energy** | `{_e:.6f} eV` |
    | **MACE reference** | `{_ref:.6f} eV` |
    | **Difference** | `{_diff:.2e} eV` |
    | **Inference time** | `{_ms} ms` |
    """)
        if _e != 0.0
        else mo.md("*Waiting for ONNX inference...*")
    )

    mo.vstack([
        mo.md("## MACE-MP-0 · ONNX Browser Demo"),
        mo.md("ML force field running via ONNX Runtime Web — no PyTorch, no server."),
        mol_select,
        mace_widget,
        _energy_row,
    ])
    return


if __name__ == "__main__":
    app.run()
