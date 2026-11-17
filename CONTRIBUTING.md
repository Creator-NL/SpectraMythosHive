# Contributing to Spectra Mythos Hive

Thank you for helping make **Spectra Mythos Hive** better! Whether you are fixing a bug on 8 GB GPUs, adding dataset recipes, improving diagnostics, or refining documentation, contributions are welcome.

---

## Development Setup

```bash
git clone https://github.com/Creator-NL/SpectraMythosHive.git
cd SpectraMythosHive
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or cu126 / cu128 for GPU
pip install -r requirements.txt
```

Run the syntax check and smoke tests locally before opening a pull request:

```bash
python -m py_compile hive_common.py hive_server.py hive_worker.py hive_worker_phase2.py test_model.py test_model_phase2.py Tools/eval_mythos.py Tools/checkpoint_diff.py Tools/plot_training_log.py
python -m unittest discover -s tests -v
```

---

## Contribution Guidelines

1. **Protect Consumer Hardware Compatibility**
   - Keep workers runnable on **8 GB VRAM GPUs** (like the **RTX 4060**) via `map_location="cpu"` checkpoint staging, native `bfloat16` initialization, and automatic `MICRO_BATCH` selection in [`hive_common.py`](hive_common.py).
   - Keep [`hive_server.py`](hive_server.py) strictly CPU-RAM based (`0 MB` GPU VRAM).

2. **Preserve Safety & Sanitization Standards**
   - Never change network deserialization away from `torch.load(..., weights_only=True)`.
   - Never bypass the `NaN` / `Inf`, tensor-type, or shape checks in `submit_gradients`.
   - Never commit personal paths (`C:\Users\...`), real passwords, `.env` files, `hive_state/`, or `.pt` weights.

3. **Preserve Upstream Attribution**
   - The `OpenMythos/` package is included under the MIT License (Copyright (c) 2026 Kye Gomez / The Swarm Corporation). Keep upstream attribution and license files intact.

4. **Be Honest in Console Output & Documentation**
   - Avoid silently hiding meaningful library logs/warnings when documenting or tuning worker output—prefer clear user-facing documentation over masking diagnostics.

5. **Zero Unwanted Telemetry**
   - Never add analytics, tracking beacons, or phone-home requests. All network traffic must remain limited to user-triggered Hugging Face asset downloads and the user's own configured `HIVE_URL`.

---

## Pull Request Process

1. Fork the repository and create a focused branch (`fix/...` or `feat/...`).
2. Add or update unit tests in `tests/` when modifying server, worker, config, or diagnostic logic.
3. Update `README.md` or `Tools/` documentation if CLI flags or `.env` variables change.
4. Ensure CI passes cleanly across Windows and Linux checks.

For questions or major architecture proposals, open a GitHub Issue or reach out at **contact@biohazard-corp.nl**.
