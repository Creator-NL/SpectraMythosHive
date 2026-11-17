"""Spectra Mythos Hive Server — CPU Master State & Distributed Gradient Coordinator."""

import asyncio
import os
import time

import torch
import uvicorn
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from hive_common import (
    ROOT_DIR,
    build_mythos_1b,
    ensure_utf8_stdio,
    get_env_float,
    get_env_int,
    load_dotenv,
)

ensure_utf8_stdio()
load_dotenv()

app = FastAPI(title="Spectra Mythos Hive Server")

# --- Configuration ---
DEFAULT_PASSWORD = "mythos_hive"
HIVE_PASSWORD = os.environ.get("HIVE_PASSWORD", DEFAULT_PASSWORD)
HIVE_HOST = os.environ.get("HIVE_HOST", "0.0.0.0")
HIVE_PORT = get_env_int("HIVE_PORT", 8000)
LR = get_env_float("HIVE_LR", 3e-5)
SAVE_EVERY = get_env_int("HIVE_SAVE_EVERY", 10)
SEQ_LEN = get_env_int("HIVE_SEQ_LEN", 256)

HIVE_STATE_DIR = os.path.join(ROOT_DIR, "hive_state")
GLOBAL_MODEL_PATH = os.path.join(HIVE_STATE_DIR, "global_model.pt")
WORKER_MODEL_PATH = os.path.join(HIVE_STATE_DIR, "worker_model.pt")
CHECKPOINT_DIR = os.path.join(HIVE_STATE_DIR, "checkpoints")

# --- Global State (held in CPU RAM so the server never wastes GPU VRAM) ---
model = None
optimizer = None
global_step = 0
dispatched_jobs = 0
pending_jobs = {}
hive_lock = asyncio.Lock()


def init_server() -> None:
    global model, optimizer, global_step, dispatched_jobs, pending_jobs
    os.makedirs(HIVE_STATE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if HIVE_PASSWORD == DEFAULT_PASSWORD:
        print("⚠️ Using default HIVE_PASSWORD ('mythos_hive'). Set HIVE_PASSWORD in .env before hosting publicly!")

    print("🐝 Initializing Hive Master Model (CPU)...")
    model = build_mythos_1b(seq_len=SEQ_LEN, vocab_size=50257)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1
    )

    if os.path.exists(GLOBAL_MODEL_PATH):
        print("📥 Loading existing Hive state...")
        # Local trusted server checkpoint holds AdamW optimizer state alongside model tensors.
        checkpoint = torch.load(GLOBAL_MODEL_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        global_step = int(checkpoint["global_step"])
        dispatched_jobs = int(checkpoint.get("dispatched_jobs", global_step))
        pending_jobs = dict(checkpoint.get("pending_jobs", {}))
        del checkpoint
        print(
            f"✅ Resumed at global step {global_step} "
            f"(Dispatched: {dispatched_jobs}, Pending: {len(pending_jobs)})"
        )
    else:
        print("🌱 Created fresh 1B model from scratch.")
        dispatched_jobs = 0
        pending_jobs = {}
        save_state()


def save_state() -> None:
    print("💾 Saving Master Model...")

    # 1. Full ~13 GB checkpoint (weights + AdamW moments) for server resume
    tmp_global = GLOBAL_MODEL_PATH + ".tmp"
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_step": global_step,
        "dispatched_jobs": dispatched_jobs,
        "pending_jobs": pending_jobs,
    }
    torch.save(state, tmp_global)
    os.replace(tmp_global, GLOBAL_MODEL_PATH)

    # 2. Lightweight ~4 GB weights-only file for workers (safe with weights_only=True)
    tmp_worker = WORKER_MODEL_PATH + ".tmp"
    torch.save({"model_state": model.state_dict()}, tmp_worker)
    os.replace(tmp_worker, WORKER_MODEL_PATH)


def verify_auth(x_hive_password: str = Header(None)) -> None:
    if x_hive_password != HIVE_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid Hive Password")


@app.get("/")
def read_root():
    return {"status": "Hive is alive", "global_step": global_step}


@app.get("/api/get_job")
async def get_job(x_hive_password: str = Header(None)):
    verify_auth(x_hive_password)
    global dispatched_jobs, pending_jobs

    current_time = time.time()
    job_id = None

    # 1. Re-assign jobs whose workers timed out (> 10 minutes)
    for pending_id, start_time in list(pending_jobs.items()):
        if current_time - start_time > 600:
            print(f"⚠️ Re-assigning dropped job #{pending_id}")
            job_id = pending_id
            pending_jobs[pending_id] = current_time
            break

    # 2. Otherwise hand out a brand-new job
    if job_id is None:
        job_id = dispatched_jobs
        dispatched_jobs += 1
        pending_jobs[job_id] = current_time

    job_size = 8192
    dataset_offset = job_id * job_size
    return {
        "job_id": job_id,
        "dataset_offset": dataset_offset,
        "job_steps": 50,
    }


@app.get("/api/download_model")
def download_model(x_hive_password: str = Header(None)):
    verify_auth(x_hive_password)
    if os.path.exists(WORKER_MODEL_PATH):
        return FileResponse(WORKER_MODEL_PATH, filename="worker_model.pt")
    return FileResponse(GLOBAL_MODEL_PATH, filename="global_model.pt")


@app.post("/api/submit_gradients")
async def submit_gradients(
    file: UploadFile = File(...),
    x_hive_password: str = Header(None),
    x_worker_name: str = Header("Unknown"),
    x_job_id: str = Header("-1"),
    x_job_size: str = Header("Unknown"),
):
    verify_auth(x_hive_password)
    global global_step, pending_jobs

    temp_path = os.path.join(HIVE_STATE_DIR, f"temp_grads_{time.time_ns()}.pt")
    with open(temp_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    try:
        # Strict weights_only=True prevents arbitrary pickle execution on network uploads.
        client_grads = torch.load(temp_path, map_location="cpu", weights_only=True)
        if not isinstance(client_grads, dict):
            raise HTTPException(status_code=400, detail="Gradient payload must be a dict of tensors.")

        async with hive_lock:
            print(
                f"⚡ Received gradient payload ({x_job_size} texts) from {x_worker_name}! "
                f"Applying to Master Model... (Step {global_step})"
            )

            optimizer.zero_grad()

            # Reject NaN / Inf payloads before touching master weights
            for name, grad in client_grads.items():
                if grad is not None:
                    if not isinstance(grad, torch.Tensor):
                        raise HTTPException(status_code=400, detail=f"Invalid non-tensor entry: {name}")
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        print(f"☠️ REJECTED: Worker {x_worker_name} sent NaN/Inf gradients in {name}!")
                        raise HTTPException(
                            status_code=400,
                            detail="Gradients contain NaN/Inf. Payload rejected to protect the Hive.",
                        )

            for name, param in model.named_parameters():
                if name in client_grads and client_grads[name] is not None:
                    grad_tensor = client_grads[name]
                    if grad_tensor.shape != param.shape:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Shape mismatch on {name}: expected {tuple(param.shape)}, got {tuple(grad_tensor.shape)}",
                        )
                    param.grad = grad_tensor.to(torch.float32)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1

            try:
                job_id_int = int(x_job_id)
                pending_jobs.pop(job_id_int, None)
            except ValueError:
                pass

            if global_step % SAVE_EVERY == 0:
                save_state()

            return {"status": "success", "global_step": global_step}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error applying gradients from {x_worker_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to apply gradients")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/report_error")
async def report_error(
    request: Request,
    x_hive_password: str = Header(None),
    x_worker_name: str = Header("Unknown"),
):
    verify_auth(x_hive_password)
    data = await request.json()
    error_msg = data.get("error", "Unknown error")
    print(f"⚠️ ERROR REPORT from {x_worker_name}: {error_msg}")
    return {"status": "acknowledged"}


if __name__ == "__main__":
    init_server()
    print(f"\n🚀 HIVE SERVER RUNNING ON {HIVE_HOST}:{HIVE_PORT}")
    print("   Friends can connect using your LAN/public IP address and shared HIVE_PASSWORD!")
    uvicorn.run(app, host=HIVE_HOST, port=HIVE_PORT)
