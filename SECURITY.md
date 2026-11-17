# Security Policy

## Supported Versions

Only the latest release on the `main` branch of `SpectraMythosHive` is actively supported with security updates.

---

## Security Architecture & Built-In Safeguards

`SpectraMythosHive` is designed for **self-hosted distributed training** across machines you own or share with trusted collaborators. Key safeguards built into the codebase include:

1. **Restricted Network Deserialization (`weights_only=True`)**
   - Worker gradient uploads received by `POST /api/submit_gradients` in [`hive_server.py`](hive_server.py) and `worker_model.pt` checkpoints loaded by workers in [`hive_common.py`](hive_common.py) are loaded strictly with `torch.load(..., weights_only=True)` to block arbitrary Python `pickle` execution over HTTP.
   - `hive_state/global_model.pt` stores local `AdamW` optimizer states (`weights_only=False` on the host machine that generated it) and is never sent to workers—workers download the stripped weights-only `worker_model.pt`. Diagnostic tools ([`Tools/eval_mythos.py`](Tools/eval_mythos.py) and [`Tools/checkpoint_diff.py`](Tools/checkpoint_diff.py)) default to `weights_only=True` unless `--allow-full-checkpoint` is passed explicitly for trusted local `global_model*.pt` files.

2. **Gradient Poison & Shape Validation**
   - `hive_server.py` checks every uploaded entry is a finite `torch.Tensor` (`torch.isnan` / `torch.isinf`), verifies exact tensor shapes against the master model, rejects invalid payloads with `HTTP 400`, and applies `torch.nn.utils.clip_grad_norm_(..., max_norm=1.0)` under an `asyncio.Lock`.
   - Workers also scan locally computed gradients before upload and report errors cleanly without crashing the server.

3. **Credential Hygiene**
   - Passwords, LAN/public Hive URLs, and worker identities live in `.env` (gitignored by [`.gitignore`](.gitignore)), never committed to source control.
   - Change `HIVE_PASSWORD` from its example placeholder before binding `hive_server.py` beyond `127.0.0.1`.

4. **Transport Security Recommendation**
   - Plain HTTP sends `x-hive-password` headers unencrypted. When sharing a Hive server across the internet with remote friends, always connect over a private mesh VPN (e.g., Tailscale or WireGuard) or terminate TLS/HTTPS behind a reverse proxy.

---

## Reporting a Vulnerability

If you discover a security vulnerability (for example in checkpoint loading, FastAPI header auth, upload streaming, or path handling):

1. **Please do NOT open a public GitHub issue** for unpatched vulnerabilities.
2. Report it privately via **GitHub Security Advisories** on this repository, or email **contact@biohazard-corp.nl** with steps to reproduce and affected file/line details.

We will acknowledge receipt promptly and work with you on a coordinated fix before public disclosure.
