# Privacy & Network Transparency

**Spectra Mythos Hive** is built for **100% self-hosted, user-owned AI training**. Biohazard Corp does **not** operate any central server, analytics pipeline, crash reporter, or hidden account system connected to this repository.

## What Network Activity Occurs?

By default, software in this repository performs **only** the network requests you explicitly initiate:

1. **Hugging Face Asset Downloads (Initial Setup / First Run)**
   - Running a worker or evaluator loads the GPT-2 tokenizer (`gpt2` / `openai-community/gpt2`) and, in Phase 1 / Phase 2, the training dataset configured in `.env` (by default `HuggingFaceFW/fineweb-edu` `sample-10BT` or `tatsu-lab/alpaca`) via the `transformers` and `datasets` libraries into your local Hugging Face cache (`~/.cache/huggingface`).
   - Once cached locally, workers and testers operate cleanly using your local disk cache and your local Hive server. (You can also point `HIVE_DATASET_SUBSET` / `HIVE_PHASE2_DATASET` to purely local datasets.)

2. **Self-Hosted Hive Server ↔ Worker Traffic (`HIVE_URL`)**
   - Workers communicate exclusively with the `HIVE_URL` configured in your local `.env` (defaulting to `http://127.0.0.1:8000` on your own machine).
   - Over that connection, workers fetch job metadata (`/api/get_job`), download weights-only checkpoints (`/api/download_model`), upload tensor gradients (`/api/submit_gradients`), and report math-overflow warnings (`/api/report_error`).
   - No prompt data, model weights, or IP addresses are ever sent to Biohazard Corp.

3. **What Stays Strictly Local on Disk**
   - `.env` (server URL, password, worker name)
   - `hive_state/global_model.pt` and `hive_state/worker_model.pt`
   - Temporary files (`hive_worker_model.pt`, `temp_upload_grads.pt`, which is deleted right after upload)
   - Console logs and plots generated locally (`training_progress.png`)

## Remote Friend Workers & Encryption

If you expose your own `hive_server.py` to trusted friends beyond your home LAN, remember that HTTP traffic transmits headers (including `x-hive-password`) unencrypted unless wrapped in a private mesh VPN (Tailscale / WireGuard) or HTTPS reverse proxy (see [`SECURITY.md`](../SECURITY.md)).
