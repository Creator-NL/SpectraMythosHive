"""Shared configuration, model initialization, and worker loop utilities."""

import contextlib
import os
import sys
import time
import warnings
from typing import Callable, Iterator, Optional, Tuple

import requests
import torch
from torch import nn
from transformers import GPT2TokenizerFast

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OPENMYTHOS_DIR = os.path.join(ROOT_DIR, "OpenMythos")
if os.path.isdir(OPENMYTHOS_DIR) and OPENMYTHOS_DIR not in sys.path:
    sys.path.insert(0, OPENMYTHOS_DIR)

from open_mythos.main import OpenMythos  # noqa: E402
from open_mythos.variants import mythos_1b  # noqa: E402


def ensure_utf8_stdio() -> None:
    """Reconfigure Windows CMD/PowerShell streams to UTF-8 so emojis never crash logs."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and getattr(stream, "encoding", "").lower() != "utf-8":
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def load_dotenv(env_path: Optional[str] = None) -> None:
    """Load key=value pairs from `.env` without overwriting existing env vars."""
    target = env_path or os.path.join(ROOT_DIR, ".env")
    if not os.path.isfile(target):
        return
    with open(target, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)


def get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"⚠️ Invalid integer for {name}='{raw}', falling back to {default}.")
        return default


def get_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"⚠️ Invalid float for {name}='{raw}', falling back to {default}.")
        return default


def build_mythos_1b(seq_len: int = 256, vocab_size: int = 50257) -> OpenMythos:
    """Instantiate the OpenMythos 1B model matching the Hive checkpoint shape."""
    config = mythos_1b()
    config.vocab_size = vocab_size
    config.max_seq_len = seq_len
    return OpenMythos(config)


def resolve_worker_dtype(device: str) -> torch.dtype:
    """Use native bfloat16 on RTX 30/40/50 GPUs, float16 on older CUDA, or float32 on CPU."""
    if device != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    print("⚠️ GPU does not support bfloat16 natively; falling back to float16.")
    return torch.float16


def resolve_micro_batch(device: str) -> int:
    """Choose a VRAM-safe micro-batch size unless overridden via HIVE_MICRO_BATCH."""
    raw = os.environ.get("HIVE_MICRO_BATCH", "auto").strip().lower()
    if raw and raw != "auto":
        try:
            val = int(raw)
            if val >= 1:
                return val
        except ValueError:
            pass
        print(f"⚠️ Invalid HIVE_MICRO_BATCH='{raw}', using auto-detection.")

    if device != "cuda":
        return 1

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_name(0)
    if vram_gb >= 14.0:
        mb = 6
    elif vram_gb >= 10.0:
        mb = 2
    else:
        # 8 GB GPUs (e.g., RTX 4060, RTX 3060 Ti) stay around ~4.5-5.5 GB VRAM at micro_batch=1
        mb = 1
    print(f"🎮 Detected {gpu_name} ({vram_gb:.1f} GB VRAM) -> auto MICRO_BATCH={mb}")
    return mb


def make_amp_context(device: str, dtype: torch.dtype):
    if device == "cuda":
        return torch.amp.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def extract_model_state(checkpoint: object) -> dict:
    """Unwrap common checkpoint dict shapes cleanly."""
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint  # type: ignore[return-value]


def download_worker_model(hive_url: str, headers: dict, local_path: str) -> None:
    """Stream the 4 GB weights-only worker checkpoint from the Hive server."""
    print("📥 Downloading latest Master Model from Hive...")
    tmp_path = local_path + ".downloading"
    with requests.get(
        f"{hive_url}/api/download_model",
        headers=headers,
        stream=True,
        timeout=(15, 600),
    ) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    os.replace(tmp_path, local_path)
    print("✅ Model downloaded.")


def load_weights_into_model(model: nn.Module, path: str, strict_weights_only: bool = True) -> None:
    """Load checkpoint onto system RAM first to avoid a VRAM spike on 8 GB GPUs."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=strict_weights_only)
    state_dict = extract_model_state(checkpoint)
    model.load_state_dict(state_dict)
    del checkpoint, state_dict


def run_worker_loop(
    phase_label: str,
    build_dataset_iter: Callable[[int, GPT2TokenizerFast], Iterator[Tuple[torch.Tensor, torch.Tensor]]],
    preview_label: str = "Parallel Text",
) -> None:
    """Main distributed worker loop shared across training phases."""
    ensure_utf8_stdio()
    load_dotenv()
    warnings.filterwarnings("ignore", message=".*Token indices sequence length is longer.*")

    hive_url = os.environ.get("HIVE_URL", "http://127.0.0.1:8000").rstrip("/")
    hive_password = os.environ.get("HIVE_PASSWORD", "mythos_hive")
    worker_name = os.environ.get("WORKER_NAME", "Worker-01")
    seq_len = get_env_int("HIVE_SEQ_LEN", 256)
    grad_accum = get_env_int("HIVE_GRAD_ACCUM", 128)
    vocab_size = 50257

    device = "cuda" if torch.cuda.is_available() else "cpu"
    micro_batch = resolve_micro_batch(device)
    worker_dtype = resolve_worker_dtype(device)
    headers = {"x-hive-password": hive_password, "x-worker-name": worker_name}

    local_model_path = os.path.join(ROOT_DIR, "hive_worker_model.pt")
    temp_grad_path = os.path.join(ROOT_DIR, "temp_upload_grads.pt")

    print(f"🐝 Starting Hive Worker ({phase_label}) as '{worker_name}' on {device.upper()} ({worker_dtype})...")
    if device == "cpu":
        print("⚠️ No CUDA GPU detected. Running on CPU will be very slow.")
    print(f"🌐 Connecting to Hive Server: {hive_url}")
    print(f"⚙️  Settings: SEQ_LEN={seq_len} | MICRO_BATCH={micro_batch} | GRAD_ACCUM={grad_accum}")

    model = build_mythos_1b(seq_len=seq_len, vocab_size=vocab_size).to(worker_dtype).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    amp_ctx = make_amp_context(device, worker_dtype)

    # Query current global_step to start at a fresh dataset offset
    initial_offset = 0
    try:
        resp = requests.get(f"{hive_url}/", headers=headers, timeout=10)
        if resp.status_code == 200:
            initial_offset = resp.json().get("global_step", 0)
    except Exception:
        pass

    print("🌊 Opening continuous dataset stream...")
    data_gen = build_dataset_iter(initial_offset, tokenizer)
    last_job_id = -1

    while True:
        try:
            resp = requests.get(f"{hive_url}/api/get_job", headers=headers, timeout=15)
            resp.raise_for_status()
            job = resp.json()
            job_id = int(job["job_id"])

            if job_id > last_job_id + 5 or last_job_id == -1:
                download_worker_model(hive_url, headers, local_model_path)
                load_weights_into_model(model, local_model_path, strict_weights_only=True)
                last_job_id = job_id

            print(f"\n🚀 Starting Job #{job_id}")
            model.train()
            model.zero_grad(set_to_none=True)

            loss_accum = 0.0
            t0 = time.perf_counter()

            for micro_i in range(grad_accum):
                x_batch = []
                y_batch = []
                for idx in range(micro_batch):
                    x, y = next(data_gen)
                    if micro_i == 0:
                        try:
                            preview = tokenizer.decode(x[:15]).replace("\n", " ")
                            print(f'    📖 {preview_label} {idx + 1}: "{preview}..."')
                        except Exception:
                            pass
                    x_batch.append(x)
                    y_batch.append(y)

                x_tensor = torch.stack(x_batch).to(device, non_blocking=True)
                y_tensor = torch.stack(y_batch).to(device, non_blocking=True)

                with amp_ctx:
                    logits = model(x_tensor, n_loops=4)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, vocab_size), y_tensor.view(-1)
                    )
                    loss = loss / grad_accum

                loss.backward()
                step_loss = loss.item() * grad_accum
                loss_accum += step_loss
                print(
                    f"    ⚡ micro-batch {micro_i + 1:3d}/{grad_accum} "
                    f"| parallel_texts: {x_tensor.size(0)} | loss {step_loss:.4f}"
                )

            avg_job_loss = loss_accum / grad_accum
            print(f"📉 Job #{job_id} Average Loss: {avg_job_loss:.4f}")
            print("📦 Packaging gradients...")

            grads = {}
            has_nan = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    g = param.grad.detach().cpu()
                    if torch.isnan(g).any() or torch.isinf(g).any():
                        has_nan = True
                        break
                    grads[name] = g

            if has_nan:
                print("☠️ ERROR: Math overflow produced NaN/Inf gradients. Discarding job to protect the Hive.")
                try:
                    requests.post(
                        f"{hive_url}/api/report_error",
                        headers=headers,
                        json={"error": "Math overflow produced NaN/Inf gradients. Job discarded."},
                        timeout=10,
                    )
                except Exception:
                    pass
                continue

            try:
                torch.save(grads, temp_grad_path)
                del grads
                print(f"📤 Uploading results for Job #{job_id} (Took {time.perf_counter() - t0:.1f}s)")
                with open(temp_grad_path, "rb") as f:
                    job_size = grad_accum * micro_batch
                    upload_headers = {
                        **headers,
                        "x-job-id": str(job_id),
                        "x-job-size": str(job_size),
                    }
                    upload_resp = requests.post(
                        f"{hive_url}/api/submit_gradients",
                        headers=upload_headers,
                        files={"file": f},
                        timeout=(15, 600),
                    )
                    upload_resp.raise_for_status()
                print(f"✅ Job #{job_id} Accepted! Global Step is now {upload_resp.json()['global_step']}")
            finally:
                if os.path.exists(temp_grad_path):
                    os.remove(temp_grad_path)

        except Exception as e:
            print(f"⚠️ Error communicating with Hive or processing job: {e}")
            print("Retrying in 10 seconds...")
            time.sleep(10)


def resolve_default_model_path() -> Optional[str]:
    """Locate the first available local checkpoint for testers and evaluation tools."""
    candidates = [
        os.path.join(ROOT_DIR, "hive_state", "worker_model.pt"),
        os.path.join(ROOT_DIR, "hive_state", "global_model.pt"),
        os.path.join(ROOT_DIR, "hive_worker_model.pt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def run_interactive_tester(
    title: str,
    prompt_formatter: Optional[Callable[[str], str]] = None,
    stop_tokens: Tuple[str, ...] = (),
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    print_prompt: bool = True,
) -> None:
    """Shared interactive CLI loop for Phase 1 completion & Phase 2 chat testing."""
    ensure_utf8_stdio()
    load_dotenv()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = resolve_default_model_path()

    print("=" * 50)
    print(f"  {title}")
    print("=" * 50)

    if model_path is None:
        print("❌ Could not find hive_state/worker_model.pt or hive_state/global_model.pt!")
        print("Start the Hive Server (StartHive.bat) once first so it creates an initial checkpoint.")
        sys.exit(1)

    print("📚 Loading Tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")

    print("🧠 Initializing OpenMythos 1B Model...")
    seq_len = get_env_int("HIVE_SEQ_LEN", 256)
    test_dtype = resolve_worker_dtype(device)
    model = build_mythos_1b(seq_len=seq_len, vocab_size=50257).to(test_dtype).to(device)

    print(f"📥 Loading weights from {os.path.relpath(model_path, ROOT_DIR)}...")
    strict_weights_only = not os.path.basename(model_path).startswith("global_model")
    load_weights_into_model(model, model_path, strict_weights_only=strict_weights_only)
    model.eval()

    print("✅ Model loaded successfully!\n")

    amp_ctx = make_amp_context(device, test_dtype)

    while True:
        try:
            user_prompt = input("👤 You: ")
            if user_prompt.strip().lower() in ("quit", "exit"):
                break
            if not user_prompt.strip():
                continue

            model_input = prompt_formatter(user_prompt) if prompt_formatter else user_prompt
            tokens = tokenizer.encode(model_input)
            if len(tokens) >= seq_len:
                tokens = tokens[-(seq_len - 1) :]
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

            print("\n🤖 AI: ", end="", flush=True)
            if print_prompt:
                print(user_prompt, end="", flush=True)

            with torch.no_grad():
                for _ in range(max_new_tokens):
                    if input_ids.shape[1] >= seq_len:
                        break
                    with amp_ctx:
                        logits = model(input_ids, n_loops=4)
                    next_logits = logits[0, -1, :] / temperature
                    top_k = min(40, next_logits.size(-1))
                    v, _ = torch.topk(next_logits, top_k)
                    next_logits[next_logits < v[[-1]]] = -float("inf")

                    probs = torch.nn.functional.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

                    word = tokenizer.decode(next_token.item())
                    if any(tok in word for tok in stop_tokens):
                        break
                    print(word, end="", flush=True)

            print("\n")
        except KeyboardInterrupt:
            break

    print("\n👋 Exiting tester. Have a great day!")
