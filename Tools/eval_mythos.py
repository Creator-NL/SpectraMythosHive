"""10-Domain Diagnostic Report Card & Live Thought Sampler for OpenMythos 1B."""

import argparse
import os
import sys
import warnings
from typing import Optional

warnings.filterwarnings("ignore")

import torch
from transformers import GPT2TokenizerFast

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_DIR)
if MAIN_DIR not in sys.path:
    sys.path.insert(0, MAIN_DIR)

from hive_common import (  # noqa: E402
    build_mythos_1b,
    ensure_utf8_stdio,
    get_env_int,
    load_dotenv,
    load_weights_into_model,
    make_amp_context,
    resolve_default_model_path,
    resolve_worker_dtype,
)

ensure_utf8_stdio()
load_dotenv()

TEST_DOMAINS = {
    "🔬 Science & Biology": (
        "Chondroitin sulfate consists of repeating chains of molecules called "
        "glycosaminoglycans. It is a major component of extracellular matrix, and is "
        "important in maintaining the structural integrity of the tissue."
    ),
    "💻 Computer Code (Python)": (
        "def calculate_average(numbers):\n"
        "    total = sum(numbers)\n"
        "    count = len(numbers)\n"
        "    if count == 0:\n"
        "        return 0\n"
        "    return total / count\n"
    ),
    "📚 History & Literature": (
        "The French viscount Urban V, born Guillaume de Grimoard, was the head of the "
        "Catholic Church from 1362 to 1370. He was the sixth Avignon pope and the only "
        "one to be beatified."
    ),
    "🧮 Math & Logic": (
        "If you have 10 apples and give 3 to a friend, you have 7 apples left. The "
        "mathematical equation for this is 10 - 3 = 7. Next, if you multiply that by 2, "
        "you get 14."
    ),
    "🤖 Casual Chat (Phase 2)": (
        "<|system|>You are a helpful assistant.\n"
        "<|user|>Hello, how are you today?\n"
        "<|ai|>I am doing well, thank you for asking! How can I help you?"
    ),
    "🌍 Foreign Language (French)": (
        "La tour Eiffel est une tour de fer puddlé de 330 mètres de hauteur située à "
        "Paris, à l'extrémité nord-ouest du parc du Champ-de-Mars en bordure de la Seine."
    ),
    "🍳 Everyday Knowledge": (
        "To make a classic omelette, whisk three eggs with a pinch of salt and pepper. "
        "Heat a tablespoon of butter in a non-stick pan over medium-high heat until it "
        "begins to foam."
    ),
    "🤔 Philosophy & Abstract": (
        "Epistemology is the branch of philosophy concerned with knowledge. "
        "Epistemologists study the nature, origin, and scope of knowledge, epistemic "
        "justification, the rationality of belief, and various related issues."
    ),
    "📝 Formatting & Markdown": (
        "Here is a list of tasks:\n"
        "1. Clean the kitchen\n"
        "2. Walk the dog\n"
        "3. Buy groceries\n\n"
        "* Remember to bring the reusable bags!"
    ),
    "🐉 Fantasy & Lore": (
        "The ancient crystal glowed with a faint, pulsating blue light. Legend said that "
        "whoever touched the artifact would be granted visions of the forgotten realm of "
        "Spectra."
    ),
}


def rate_loss(loss_val: float) -> str:
    if loss_val < 2.8:
        return "⭐⭐⭐⭐⭐ Master"
    if loss_val < 3.2:
        return "⭐⭐⭐⭐ Advanced"
    if loss_val < 3.8:
        return "⭐⭐⭐ Proficient"
    if loss_val < 4.8:
        return "⭐⭐ Developing"
    if loss_val < 6.0:
        return "⭐ Novice"
    return "❌ Untrained"


def evaluate_model(
    model_path: Optional[str] = None,
    seq_len: int = 256,
    n_loops: int = 4,
    allow_full_checkpoint: bool = False,
) -> None:
    target_path = model_path or resolve_default_model_path()
    if not target_path or not os.path.isfile(target_path):
        print("❌ Could not find a checkpoint (checked hive_state/worker_model.pt, hive_state/global_model.pt, hive_worker_model.pt).")
        print("   Start the Hive Server once first or pass --model path/to/checkpoint.pt")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = resolve_worker_dtype(device)

    print("📥 Loading Model and Tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = build_mythos_1b(seq_len=seq_len, vocab_size=50257).to(dtype).to(device)

    strict_weights_only = (not allow_full_checkpoint) and (
        not os.path.basename(target_path).startswith("global_model")
    )
    load_weights_into_model(model, target_path, strict_weights_only=strict_weights_only)
    model.eval()

    print(f"✅ Loaded {target_path} on {device.upper()}! Running diagnostics...\n")
    print("=" * 60)
    print(f"{'TEST DOMAIN':<35} | {'LOSS':<10} | {'HEALTH'}")
    print("=" * 60)

    total_loss = 0.0
    amp_ctx = make_amp_context(device, dtype)

    with torch.no_grad():
        for domain, text in TEST_DOMAINS.items():
            tokens = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
            x = tokens[:, :-1]
            y = tokens[:, 1:]

            with amp_ctx:
                logits = model(x, n_loops=n_loops)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )

            loss_val = loss.item()
            total_loss += loss_val
            print(f"{domain:<35} | {loss_val:<10.4f} | {rate_loss(loss_val)}")

    print("=" * 60)
    print(f"OVERALL AVERAGE LOSS: {total_loss / len(TEST_DOMAINS):.4f}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("🧠 LIVE SAMPLES: WATCHING OPENMYTHOS WRITE")
    print("=" * 60)

    sample_prompts = [
        "Science has proven that",
        "To make a classic omelette,",
        "The ancient secret of the realm of Spectra is",
    ]

    with torch.no_grad():
        for prompt in sample_prompts:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            with amp_ctx:
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=25,
                    n_loops=n_loops,
                    temperature=0.75,
                    top_k=50,
                )
            gen_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
            clean_text = gen_text.replace("\n", " ").strip()
            print(f'💬 Prompt  : "{prompt}"')
            print(f"🤖 Mythos  : {clean_text}\n")

    print("=" * 60)
    print("💡 Tip: As loss drops into the 3s and 2s, writing becomes")
    print("   increasingly coherent, structured, and factual!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an OpenMythos Hive checkpoint across 10 domains.")
    parser.add_argument("--model", default=None, help="Path to a .pt checkpoint (auto-detected if omitted)")
    parser.add_argument("--seq-len", type=int, default=get_env_int("HIVE_SEQ_LEN", 256), help="Sequence length (default 256)")
    parser.add_argument("--loops", type=int, default=4, help="Recurrent loop iterations during eval (default 4)")
    parser.add_argument("--allow-full-checkpoint", action="store_true", help="Allow loading trusted local optimizer-included checkpoints")
    args = parser.parse_args()
    evaluate_model(
        model_path=args.model,
        seq_len=args.seq_len,
        n_loops=args.loops,
        allow_full_checkpoint=args.allow_full_checkpoint,
    )
