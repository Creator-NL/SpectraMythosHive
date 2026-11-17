# Mythos Training Visibility — Reference Guide

A guide to inspecting your OpenMythos Hive training runs: checking the neural structure, tracking real loss trends, comparing checkpoints, and telling the difference between "loss fluctuated" and "the model actually got smarter."

---

## 1. 10-Domain Report Card & Live Sampler — `Tools/eval_mythos.py`

Run standardized cross-entropy loss across 10 diverse domains plus live autoregressive text generation:

```bash
python Tools/eval_mythos.py
# Or inspect a specific saved checkpoint:
python Tools/eval_mythos.py --model hive_state/worker_model.pt --loops 4
```

* Lower loss means less confusion (`< 2.8` = ⭐⭐⭐⭐⭐ Master, `< 3.2` = ⭐⭐⭐⭐ Advanced, `< 3.8` = ⭐⭐⭐ Proficient).
* Because `FineWeb-Edu` is an English educational corpus, Science and Philosophy naturally drop first while Fantasy/French remain higher until targeted Phase 1.5 / Phase 2 diets.

---

## 2. Watching the learning process — `Tools/plot_training_log.py`

Parses your Hive Worker console output and plots two charts side by side:
- Raw per-micro-batch loss (noisy by design) + 20-batch moving average
- Per-job average loss vs. global step (the true training trend)

Save your worker output to a file:

```bash
python hive_worker.py > training.log
python Tools/plot_training_log.py training.log
```

Output is saved to `training_progress.png`.

---

## 3. Seeing which layers changed — `Tools/checkpoint_diff.py`

Compares any two `.pt` checkpoints layer-by-layer (all 276 tensors in OpenMythos 1B) to prove the weights are genuinely progressing and inspect which MoE experts are specializing:

```bash
python Tools/checkpoint_diff.py backups/step_1000.pt hive_state/worker_model.pt
python Tools/checkpoint_diff.py backups/step_1000.pt hive_state/worker_model.pt --top 300
```

- `delta norm` — absolute Euclidean movement (`||W_new - W_old||_2`).
- `delta %` — movement relative to that tensor's own magnitude.
- `freqs_cis` and `freqs_cis_mla` should always show `0.000%` (frozen positional clocks).
- Combining `checkpoint_diff.py` (movement) with `eval_mythos.py` (falling loss) proves forward progress rather than **Gradient Oscillation**.

---

## 4. Visualizing the computation graph — Netron & `torchinfo`

* **Netron:** Drag `hive_state/worker_model.pt` onto [netron.app](https://netron.app) for an interactive tensor-shape diagram.
* **`torchinfo`:** `pip install torchinfo` lets you inspect layer dimensions and exact parameter counts in Python.

---

## 5. Understanding Perplexity (`math.exp(loss)`)

Perplexity turns cross-entropy loss into intuitive guessing difficulty:
- Loss `10.84` ≈ Perplexity `51,000` (pure random guessing over GPT-2's 50,257-token vocabulary)
- Loss `4.37` ≈ Perplexity `79`
- Loss `3.00` ≈ Perplexity `20`
- Loss `2.70` ≈ Perplexity `15`

---

## 6. Understanding Console Messages

### `[transformers] Token indices sequence length is longer than the specified maximum sequence length... (3148 > 1024)`

You will see this notice on startup or during long articles in `hive_worker.py` (kept intentionally visible rather than masked). It is **expected and completely harmless**:
- Hugging Face's GPT-2 tokenizer defaults to reporting lengths over `1024` tokens when handed a full article (`truncation=False`).
- Immediately afterward, `hive_worker.py` slices `token_buffer` into fixed `SEQ_LEN` chunks (`256` tokens by default) before feeding tensors to `OpenMythos` (`max_seq_len=256`), so the model itself never receives an oversized sequence or hits indexing errors.

