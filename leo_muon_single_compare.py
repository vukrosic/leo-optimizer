#!/usr/bin/env python3
"""
Compare single runs:
- Leo at LR=1e-4
- Muon at default LR from llm_muon.py

Test run: 300 steps, eval every 100 steps.
Generates two plots:
- Validation Loss vs Training Steps (overlay)
- Validation Loss vs Wall-clock Time (overlay)
"""

import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import leo_vs_muon_lr_ablation as ab


def fetch_time_axis(metrics: dict, fallback_points: int, eval_every: int, max_steps: int):
    # Prefer real eval timestamps recorded by train_with_lr; otherwise approximate
    wt = metrics.get('eval_wall_times')
    if isinstance(wt, list) and len(wt) > 0:
        return wt
    total_time = metrics.get('training_time', None)
    if total_time is None or max_steps == 0:
        return [i * eval_every for i in range(fallback_points)]
    tps = total_time / max_steps
    return [i * eval_every * tps for i in range(fallback_points)]


def plot_overlays(step_axis_data, time_axis_data, out_prefix: str):
    # step-axis overlay
    plt.figure(figsize=(9, 5))
    for label, (steps, vals) in step_axis_data.items():
        plt.plot(steps, vals, marker='o', linewidth=2, label=label)
    plt.title('Validation Loss vs Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Validation Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fn_steps = f"{out_prefix}_loss_vs_steps.png"
    plt.savefig(fn_steps, dpi=300, bbox_inches='tight')
    print(f"📈 Saved plot: {fn_steps}")

    # time-axis overlay
    plt.figure(figsize=(9, 5))
    for label, (times, vals) in time_axis_data.items():
        plt.plot(times, vals, marker='o', linewidth=2, label=label)
    plt.title('Validation Loss vs Time (seconds)')
    plt.xlabel('Time (s)')
    plt.ylabel('Validation Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fn_time = f"{out_prefix}_loss_vs_time.png"
    plt.savefig(fn_time, dpi=300, bbox_inches='tight')
    print(f"⏱️ Saved plot: {fn_time}")

    return fn_steps, fn_time


def main():
    print("🧪 Leo vs Muon single-run compare (test: 300 steps)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Config
    config = ab.LRAblationConfig()
    config.max_steps = 5000
    config.eval_every = 100

    # Data
    texts, tokenizer, tokens = ab.load_and_cache_data(config)
    dataset = ab.TextTokenDataset(tokens, config.max_seq_len)

    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # LRs
    muon_lr = ab.read_muon_default_lr()
    leo_lr = 1e-4
    print(f"🔵 Muon LR: {muon_lr}")
    print(f"🔴 Leo LR: {leo_lr}")

    # Run Muon
    m_train, m_val, m_metrics = ab.train_with_lr(
        optimizer_name='Muon', main_lr=muon_lr, config=config,
        train_loader=train_loader, val_loader=val_loader
    )

    # Run Leo
    l_train, l_val, l_metrics = ab.train_with_lr(
        optimizer_name='Leo', main_lr=leo_lr, config=config,
        train_loader=train_loader, val_loader=val_loader
    )

    # Build axes
    steps_idx = [i * config.eval_every for i in range(len(m_val))]
    step_axis = {
        f"Muon (LR={muon_lr})": (steps_idx, m_val),
        f"Leo (LR={leo_lr})": (steps_idx, l_val),
    }

    muon_times = fetch_time_axis(m_metrics, len(m_val), config.eval_every, config.max_steps)
    leo_times = fetch_time_axis(l_metrics, len(l_val), config.eval_every, config.max_steps)
    time_axis = {
        f"Muon (LR={muon_lr})": (muon_times, m_val),
        f"Leo (LR={leo_lr})": (leo_times, l_val),
    }

    # Plots
    prefix = f"leo_muon_single_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    fn_steps, fn_time = plot_overlays(step_axis, time_axis, prefix)

    # Save JSON summary
    payload = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_steps": config.max_steps,
            "eval_every": config.eval_every,
            "batch_size": config.batch_size,
            "adamw_lr": config.adamw_lr,
            "num_documents": config.num_documents,
            "max_tokens": config.max_tokens,
        },
        "muon": {"lr": muon_lr, "metrics": m_metrics},
        "leo": {"lr": leo_lr, "metrics": l_metrics},
        "plots": {"loss_vs_steps": fn_steps, "loss_vs_time": fn_time},
    }
    json_fn = f"{prefix}.json"
    with open(json_fn, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"📄 Saved summary: {json_fn}")


if __name__ == '__main__':
    main()


