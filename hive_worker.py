"""Spectra Mythos Hive Worker (Phase 1 — FineWeb-Edu Pre-Training)."""

import os
import zlib

import torch
from datasets import load_dataset

from hive_common import get_env_int, load_dotenv, run_worker_loop

load_dotenv()

DATASET_SUBSET = os.environ.get("HIVE_DATASET_SUBSET", "sample-10BT")
SEQ_LEN = get_env_int("HIVE_SEQ_LEN", 256)
WORKER_NAME = os.environ.get("WORKER_NAME", "Worker-01")


def get_dataset_iterator(initial_step: int, tokenizer):
    """Download/load FineWeb-Edu locally and shuffle uniquely per worker & session."""
    raw_offset = initial_step * 768
    worker_mix = zlib.crc32(WORKER_NAME.encode("utf-8")) & 0xFFFF
    shuffle_seed = (raw_offset ^ worker_mix) % 9_500_000

    print(f"📚 Loading FineWeb-Edu ({DATASET_SUBSET}) from local cache (shuffle seed: {shuffle_seed:,})...")
    print("   (First run downloads ~27 GB into ~/.cache/huggingface/datasets. Subsequent runs load from disk.)")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=DATASET_SUBSET,
        split="train",
    )
    dataset = dataset.shuffle(seed=shuffle_seed)
    print(f"✅ Dataset ready! {len(dataset):,} articles shuffled with seed {shuffle_seed:,}.")

    def data_generator():
        while True:
            token_buffer = []
            for row in dataset:
                tokens = tokenizer(row["text"], truncation=False)["input_ids"]
                token_buffer.extend(tokens)

                while len(token_buffer) >= SEQ_LEN + 1:
                    chunk = token_buffer[: SEQ_LEN + 1]
                    token_buffer = token_buffer[SEQ_LEN:]
                    yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(
                        chunk[1:], dtype=torch.long
                    )

            print("🔄 Completed a full pass over the dataset! Starting next pass...")

    return data_generator()


if __name__ == "__main__":
    run_worker_loop(
        phase_label="PHASE 1 - PRE-TRAINING",
        build_dataset_iter=get_dataset_iterator,
        preview_label="Parallel Text",
    )
