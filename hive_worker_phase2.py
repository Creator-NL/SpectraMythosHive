"""Spectra Mythos Hive Worker (Phase 2 — Instruction & Companion Dialogue Template)."""

import os

import torch
from datasets import load_dataset

from hive_common import get_env_int, load_dotenv, run_worker_loop

load_dotenv()

PHASE2_DATASET = os.environ.get("HIVE_PHASE2_DATASET", "tatsu-lab/alpaca")
SYSTEM_PROMPT = os.environ.get(
    "HIVE_SYSTEM_PROMPT",
    "You are a helpful, honest AI companion. Think carefully and converse naturally.",
)
SEQ_LEN = get_env_int("HIVE_SEQ_LEN", 256)


def get_dataset_iterator(initial_step: int, tokenizer):
    """Stream conversational SFT samples in `<|system|>/<|user|>/<|ai|>` format."""
    print(f"📚 Loading Phase 2 SFT dataset ({PHASE2_DATASET})...")

    def data_generator():
        base_dataset = load_dataset(PHASE2_DATASET, split="train", streaming=True)
        safe_offset = (initial_step * 8192) % 50000
        if safe_offset > 0:
            print(f"⏭️ Skipping {safe_offset:,} conversations to reduce worker overlap...")
            current_iterator = iter(base_dataset.skip(safe_offset))
        else:
            current_iterator = iter(base_dataset)

        token_buffer = []
        while True:
            try:
                row = next(current_iterator)
                instruction = row.get("instruction", "")
                input_text = row.get("input", "")
                output_text = row.get("output", "")

                user_msg = f"{instruction}\n{input_text}" if input_text else instruction
                formatted_text = (
                    f"<|system|>\n{SYSTEM_PROMPT}\n"
                    f"<|user|>\n{user_msg}\n"
                    f"<|ai|>\n{output_text}<|endoftext|>"
                )

                tokens = tokenizer(formatted_text, truncation=False)["input_ids"]
                token_buffer.extend(tokens)

                while len(token_buffer) >= SEQ_LEN + 1:
                    chunk = token_buffer[: SEQ_LEN + 1]
                    token_buffer = token_buffer[SEQ_LEN:]
                    yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(
                        chunk[1:], dtype=torch.long
                    )
            except StopIteration:
                print("🔄 Dataset reached the end! Looping back to conversation #1...")
                current_iterator = iter(base_dataset)

    return data_generator()


if __name__ == "__main__":
    run_worker_loop(
        phase_label="PHASE 2 - INTENT / SFT",
        build_dataset_iter=get_dataset_iterator,
        preview_label="Parallel Chat",
    )
