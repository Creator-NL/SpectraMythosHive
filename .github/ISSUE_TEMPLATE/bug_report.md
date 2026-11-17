---
name: Bug report
about: Report a crash, VRAM issue, Hive sync bug, or setup problem
title: "[Bug]: "
labels: bug
assignees: ''
---

**Describe the bug**
A clear description of what happened.

**Environment**
- OS: (e.g., Windows 11, Ubuntu 22.04)
- Python version: (e.g., 3.10.0)
- PyTorch + CUDA build: (e.g., `2.x+cu126`, `2.x+cu128`, or CPU)
- GPU & VRAM: (e.g., RTX 4060 8 GB, RTX 5060 Ti 16 GB)
- System RAM: (e.g., 8 GB, 32 GB)
- Component: (Hive Server / Phase 1 Worker / Phase 2 Worker / Eval / Setup)

**`.env` Settings (NEVER paste your real `HIVE_PASSWORD`!)**
- `HIVE_MICRO_BATCH`: (`auto` / `1` / `6`)
- `HIVE_GRAD_ACCUM`: (`128` / etc.)
- `HIVE_SEQ_LEN`: (`256`)

**Steps to reproduce**
1. Run '...'
2. See error '...'

**Console output**
Paste the exact traceback or log lines here (redact passwords/public IPs).
