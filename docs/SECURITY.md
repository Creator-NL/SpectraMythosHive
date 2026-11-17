# Security Policy (Alias)

The canonical security policy and vulnerability disclosure instructions live in [`SECURITY.md`](../SECURITY.md) at the repository root.

Key highlights for `SpectraMythosHive`:
- Network gradient payloads (`/api/submit_gradients`) and worker weights (`worker_model.pt`) use `torch.load(..., weights_only=True)` to prevent arbitrary `pickle` code execution.
- Uploaded gradients are checked for tensor types, exact parameter shapes, and `NaN`/`Inf` overflow (`HTTP 400` rejection), then clipped to `max_norm=1.0`.
- Keep `.env` secret, change `HIVE_PASSWORD`, and wrap public-internet connections in TLS or a private VPN (Tailscale / WireGuard).
- Report security issues privately via GitHub Security Advisories or **contact@biohazard-corp.nl**.
