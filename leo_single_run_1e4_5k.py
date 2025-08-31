#!/usr/bin/env python3
"""
Single Leo run: LR=1e-4 for 5000 steps
Reuses components from leo_vs_muon_lr_ablation.py
"""

import os
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import leo_vs_muon_lr_ablation as ab


def plot_val_curve(val_losses, eval_every: int, title_suffix: str = "") -> str:
    steps = list(range(0, len(val_losses) * eval_every, eval_every))
    plt.figure(figsize=(8, 5))
    plt.plot(steps, val_losses, marker='o', linewidth=2)
    plt.title(f"Leo (LR=1e-4) Validation Loss vs Steps{title_suffix}")
    plt.xlabel("Training Step")
    plt.ylabel("Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"leo_single_run_lr1e4_5000_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📈 Saved validation loss plot to {filename}")
    return filename


def save_results_json(metrics: dict, config: ab.LRAblationConfig, plot_file: str) -> str:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "max_steps": config.max_steps,
            "batch_size": config.batch_size,
            "adamw_lr": config.adamw_lr,
            "num_documents": config.num_documents,
            "max_tokens": config.max_tokens,
            "eval_every": config.eval_every,
        },
        "run": {
            "optimizer": "Leo",
            "main_lr": 1e-4,
            "metrics": metrics,
            "plot_file": plot_file,
        },
    }

    filename = f"leo_single_run_lr1e4_5000_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"📄 Saved results JSON to {filename}")
    return filename


def main():
    print("🧪 Leo single run: LR=1e-4, 5000 steps")
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

    # Train
    train_losses, val_losses, metrics = ab.train_with_lr(
        optimizer_name="Leo", main_lr=1e-4, config=config,
        train_loader=train_loader, val_loader=val_loader
    )

    # Save outputs
    plot_file = plot_val_curve(val_losses, config.eval_every)
    save_results_json(metrics, config, plot_file)


if __name__ == "__main__":
    main()


