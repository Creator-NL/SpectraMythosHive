"""
plot_training_log.py — Turn Hive-style training console output into loss curves.

Reads a log file (or stdin) containing lines like:
    ⚡ micro-batch  1/128 | parallel_texts: 6 | loss 4.4189
    📉 Job #17497 Average Loss: 4.3705
    ✅ Job #17497 Accepted! Global Step is now 17498

and produces two plots:
  1. Raw per-micro-batch loss + moving average
  2. Per-job average loss vs. global step

Usage:
    python Tools/plot_training_log.py training.log
    cat training.log | python Tools/plot_training_log.py

Requires: matplotlib
"""
import re
import sys

import matplotlib.pyplot as plt

MICRO_BATCH_RE = re.compile(r"micro-batch\s+(\d+)/(\d+).*?loss\s+([\d.]+)")
JOB_AVG_RE = re.compile(r"Job #(\d+) Average Loss:\s+([\d.]+)")
GLOBAL_STEP_RE = re.compile(r"Job #(\d+) Accepted! Global Step is now\s+(\d+)")


def parse_log(lines):
    micro_losses = []
    job_avg_losses = []
    global_steps = {}

    for line in lines:
        m = MICRO_BATCH_RE.search(line)
        if m:
            micro_losses.append(float(m.group(3)))
            continue
        m = JOB_AVG_RE.search(line)
        if m:
            job_avg_losses.append((int(m.group(1)), float(m.group(2))))
            continue
        m = GLOBAL_STEP_RE.search(line)
        if m:
            global_steps[int(m.group(1))] = int(m.group(2))

    return micro_losses, job_avg_losses, global_steps


def moving_average(values, window=20):
    if len(values) < 2:
        return values
    window = min(window, len(values))
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i - lo + 1))
    return out


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    micro_losses, job_avg_losses, global_steps = parse_log(lines)

    if not micro_losses and not job_avg_losses:
        print("No recognizable loss lines found — check the log format / regexes.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if micro_losses:
        axes[0].plot(micro_losses, color="#999999", linewidth=0.8, label="raw micro-batch loss")
        axes[0].plot(moving_average(micro_losses, 20), color="#d62728", linewidth=2, label="20-batch moving avg")
        axes[0].set_title("Per-micro-batch loss (expect noise)")
        axes[0].set_xlabel("micro-batch #")
        axes[0].set_ylabel("loss")
        axes[0].legend()

    if job_avg_losses:
        job_ids = [j for j, _ in job_avg_losses]
        avgs = [a for _, a in job_avg_losses]
        steps = [global_steps.get(j, j) for j in job_ids]
        axes[1].plot(steps, avgs, marker="o", color="#1f77b4")
        axes[1].set_title("Per-job average loss vs. global step")
        axes[1].set_xlabel("global step")
        axes[1].set_ylabel("job average loss")

    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=150)
    print("Saved plot to training_progress.png")
    print(f"Parsed {len(micro_losses)} micro-batches and {len(job_avg_losses)} jobs.")


if __name__ == "__main__":
    main()
