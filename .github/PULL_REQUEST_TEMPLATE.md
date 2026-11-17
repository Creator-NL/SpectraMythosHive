## Summary

Describe what changed and why.

## Checklist

- [ ] No hardcoded user paths (`C:\Users\...`), passwords, `.env`, `venv/`, `hive_state/`, or `.pt` binaries included
- [ ] Preserves consumer GPU support (including 8 GB GPUs like RTX 4060 via CPU checkpoint staging & `MICRO_BATCH=1`)
- [ ] Keeps `torch.load(..., weights_only=True)` on network uploads/downloads and `NaN`/`Inf` gradient validation intact
- [ ] Keeps upstream `OpenMythos` MIT attribution intact
- [ ] Ran `python -m py_compile ...` and `python -m unittest discover -s tests -v` locally
