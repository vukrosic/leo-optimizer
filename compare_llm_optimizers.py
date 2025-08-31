#!/usr/bin/env python3
"""
LLM Optimizer Comparison Script
Compares Muon vs Leo optimizers on language modeling task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import warnings
import os
import pickle
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class ComparisonConfig:
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 24
    max_steps: int = 1000

    # Training parameters
    gradient_accumulation_steps: int = 4
    lr: float = 0.01

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 1000  # Reduced for faster comparison
    max_tokens: int = 250000   # Reduced for faster comparison

    # Evaluation
    eval_every: int = 100
    eval_steps: int = 50

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

# Import optimizers from both files
@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)

class Leo(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.01, align_const=0.2, eps=1e-8):
        """Leo (Lion with Element-wise Orthogonalization-proxy) optimizer."""
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, align_const=align_const, eps=eps)
        super(Leo, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            align_const = group['align_const']
            eps = group['eps']
            wd_factor = -lr * weight_decay

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)

                momentum = state['momentum']
                update_direction = torch.lerp(grad, momentum, beta1)
                momentum.lerp_(grad, 1 - beta2)

                if p.dim() == 2:
                    row_norm = torch.linalg.norm(update_direction, ord=2, dim=1, keepdim=True)
                    col_norm = torch.linalg.norm(update_direction, ord=2, dim=0, keepdim=True)
                    
                    update = torch.div(update_direction, row_norm.add(eps))
                    update.addcdiv_(update_direction, col_norm.add(eps))
                    
                    rms = torch.sqrt(torch.mean(update.square()) + eps)
                    scaling_factor = align_const / (rms + eps)
                    update.mul_(scaling_factor)
                    update_direction = update
                else:
                    update_direction.sign_().mul_(align_const)

                if weight_decay != 0:
                    p.add_(p, alpha=wd_factor)
                
                p.add_(update_direction, alpha=-lr)

        return loss

# Import model components (simplified versions for comparison)
def load_and_cache_data(config: ComparisonConfig, cache_dir: str = "comparison_cache"):
    """Load and cache tokenized data"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size
        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"🔄 Processing new data (will cache for future use)")
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)

    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:3000])

    print(f"Loaded {len(texts)} documents")

    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size

    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"💾 Cached data to {cache_file}")
    return texts, tokenizer, tokens

class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        Q = self.rotary(Q)
        K = self.rotary(K)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class MinimalLLM(nn.Module):
    def __init__(self, config: ComparisonConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.max_seq_len, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ComparisonConfig):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            x, y = x.to(device), y.to(device)

            with autocast(enabled=config.use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

def setup_optimizer(model: nn.Module, optimizer_name: str, config: ComparisonConfig):
    """Setup optimizer with hybrid approach"""
    main_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            main_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  {optimizer_name} parameters: {sum(p.numel() for p in main_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    if optimizer_name == "Muon":
        main_optimizer = Muon(main_params, lr=config.lr, momentum=0.95)
    elif optimizer_name == "Leo":
        main_optimizer = Leo(main_params, lr=config.lr, betas=(0.9, 0.99), align_const=0.3)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.lr*0.1, weight_decay=config.weight_decay)

    return [main_optimizer, adamw_optimizer]

def train_model_with_optimizer(optimizer_name: str, config: ComparisonConfig, 
                             train_loader: DataLoader, val_loader: DataLoader) -> Tuple[List[float], List[float], Dict]:
    """Train model with specified optimizer and return metrics"""
    print(f"\n🚀 Training model with {optimizer_name} optimizer")
    
    # Initialize model
    set_seed(42)  # Same seed for fair comparison
    model = MinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Total parameters: {total_params:,}")

    # Setup optimizers
    optimizers = setup_optimizer(model, optimizer_name, config)

    # Learning rate schedule
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    scaler = GradScaler() if config.use_amp else None

    # Training metrics
    train_losses = []
    val_losses = []
    step_times = []

    # Training loop
    model.train()
    step = 0
    start_time = time.time()

    pbar = tqdm(total=config.max_steps, desc=f"Training {optimizer_name}")

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            step_start = time.time()
            x, y = x.to(device), y.to(device)

            # Forward pass with gradient accumulation
            if config.use_amp:
                with autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

            # Optimizer step after accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

            step_time = time.time() - step_start
            step_times.append(step_time)

            # Record training loss
            current_loss = loss.item() * config.gradient_accumulation_steps
            train_losses.append(current_loss)

            # Evaluation
            if step % config.eval_every == 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                val_losses.append(eval_metrics['val_loss'])
                
                if step % (config.eval_every * 2) == 0:  # Less frequent printing
                    print(f"\nStep {step}: Train Loss: {current_loss:.4f}, "
                          f"Val Loss: {eval_metrics['val_loss']:.4f}, "
                          f"Val Acc: {eval_metrics['val_accuracy']:.4f}")

            step += 1
            pbar.update(1)

    pbar.close()

    training_time = time.time() - start_time
    print(f"  ⏱️ Training completed in {training_time:.1f} seconds")

    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    print(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, "
          f"Acc: {final_eval['val_accuracy']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")

    final_metrics = {
        'final_val_loss': final_eval['val_loss'],
        'final_val_accuracy': final_eval['val_accuracy'],
        'final_val_perplexity': final_eval['val_perplexity'],
        'training_time': training_time,
        'avg_step_time': np.mean(step_times),
        'total_params': total_params
    }

    return train_losses, val_losses, final_metrics

def plot_comparison(muon_train_losses: List[float], muon_val_losses: List[float],
                   leo_train_losses: List[float], leo_val_losses: List[float],
                   muon_metrics: Dict, leo_metrics: Dict, config: ComparisonConfig):
    """Plot comparison results"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training Loss
    ax1.plot(muon_train_losses, label='Muon', alpha=0.7, color='blue')
    ax1.plot(leo_train_losses, label='Leo', alpha=0.7, color='red')
    ax1.set_title('Training Loss Comparison')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation Loss
    eval_steps = list(range(0, len(muon_val_losses) * config.eval_every, config.eval_every))
    ax2.plot(eval_steps, muon_val_losses, label='Muon', marker='o', color='blue')
    ax2.plot(eval_steps, leo_val_losses, label='Leo', marker='s', color='red')
    ax2.set_title('Validation Loss Comparison')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training Loss (Smoothed)
    window = 50
    if len(muon_train_losses) > window:
        muon_smooth = np.convolve(muon_train_losses, np.ones(window)/window, mode='valid')
        leo_smooth = np.convolve(leo_train_losses, np.ones(window)/window, mode='valid')
        ax3.plot(muon_smooth, label='Muon (smoothed)', color='blue')
        ax3.plot(leo_smooth, label='Leo (smoothed)', color='red')
        ax3.set_title('Smoothed Training Loss')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Performance Metrics Bar Chart
    metrics_names = ['Final Val Loss', 'Final Val Accuracy', 'Training Time (s)', 'Avg Step Time (ms)']
    muon_values = [
        muon_metrics['final_val_loss'],
        muon_metrics['final_val_accuracy'],
        muon_metrics['training_time'],
        muon_metrics['avg_step_time'] * 1000
    ]
    leo_values = [
        leo_metrics['final_val_loss'],
        leo_metrics['final_val_accuracy'],
        leo_metrics['training_time'],
        leo_metrics['avg_step_time'] * 1000
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    # Normalize values for better visualization
    normalized_muon = []
    normalized_leo = []
    for i, (m_val, l_val) in enumerate(zip(muon_values, leo_values)):
        if i == 1:  # Accuracy - higher is better
            max_val = max(m_val, l_val)
            normalized_muon.append(m_val / max_val)
            normalized_leo.append(l_val / max_val)
        else:  # Loss and time - lower is better
            max_val = max(m_val, l_val)
            normalized_muon.append(max_val / m_val if m_val > 0 else 0)
            normalized_leo.append(max_val / l_val if l_val > 0 else 0)
    
    ax4.bar(x - width/2, normalized_muon, width, label='Muon', color='blue', alpha=0.7)
    ax4.bar(x + width/2, normalized_leo, width, label='Leo', color='red', alpha=0.7)
    ax4.set_title('Performance Metrics (Normalized)')
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Normalized Score (Higher = Better)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'llm_optimizer_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plot saved as {filename}")
    
    plt.show()

def save_results(muon_metrics: Dict, leo_metrics: Dict, config: ComparisonConfig):
    """Save comparison results to JSON"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'n_heads': config.n_heads,
            'max_steps': config.max_steps,
            'batch_size': config.batch_size,
            'lr': config.lr,
            'num_documents': config.num_documents,
            'max_tokens': config.max_tokens
        },
        'muon_results': muon_metrics,
        'leo_results': leo_metrics,
        'comparison': {
            'val_loss_improvement': (muon_metrics['final_val_loss'] - leo_metrics['final_val_loss']) / muon_metrics['final_val_loss'] * 100,
            'accuracy_improvement': (leo_metrics['final_val_accuracy'] - muon_metrics['final_val_accuracy']) / muon_metrics['final_val_accuracy'] * 100,
            'speed_improvement': (muon_metrics['avg_step_time'] - leo_metrics['avg_step_time']) / muon_metrics['avg_step_time'] * 100
        }
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'llm_comparison_results_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Results saved to {filename}")
    return results

def main():
    """Main comparison function"""
    print("🔬 LLM Optimizer Comparison: Muon vs Leo")
    print("=" * 60)
    
    # Check system
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create config
    config = ComparisonConfig()
    print(f"\n📋 Configuration:")
    print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    print(f"   Training: {config.max_steps} steps, batch size {config.batch_size}")
    print(f"   Data: {config.max_tokens:,} tokens from {config.num_documents} documents")
    
    # Load data
    texts, tokenizer, tokens = load_and_cache_data(config)
    dataset = TextTokenDataset(tokens, config.max_seq_len)
    
    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Train with Muon
    print("\n" + "="*60)
    muon_train_losses, muon_val_losses, muon_metrics = train_model_with_optimizer(
        "Muon", config, train_loader, val_loader
    )
    
    # Train with Leo
    print("\n" + "="*60)
    leo_train_losses, leo_val_losses, leo_metrics = train_model_with_optimizer(
        "Leo", config, train_loader, val_loader
    )
    
    # Compare results
    print("\n" + "="*60)
    print("📊 COMPARISON RESULTS")
    print("="*60)
    
    print(f"\nMuon Results:")
    print(f"  Final Val Loss: {muon_metrics['final_val_loss']:.4f}")
    print(f"  Final Val Accuracy: {muon_metrics['final_val_accuracy']:.4f}")
    print(f"  Final Val Perplexity: {muon_metrics['final_val_perplexity']:.2f}")
    print(f"  Training Time: {muon_metrics['training_time']:.1f}s")
    print(f"  Avg Step Time: {muon_metrics['avg_step_time']*1000:.1f}ms")
    
    print(f"\nLeo Results:")
    print(f"  Final Val Loss: {leo_metrics['final_val_loss']:.4f}")
    print(f"  Final Val Accuracy: {leo_metrics['final_val_accuracy']:.4f}")
    print(f"  Final Val Perplexity: {leo_metrics['final_val_perplexity']:.2f}")
    print(f"  Training Time: {leo_metrics['training_time']:.1f}s")
    print(f"  Avg Step Time: {leo_metrics['avg_step_time']*1000:.1f}ms")
    
    # Calculate improvements
    val_loss_improvement = (muon_metrics['final_val_loss'] - leo_metrics['final_val_loss']) / muon_metrics['final_val_loss'] * 100
    accuracy_improvement = (leo_metrics['final_val_accuracy'] - muon_metrics['final_val_accuracy']) / muon_metrics['final_val_accuracy'] * 100
    speed_improvement = (muon_metrics['avg_step_time'] - leo_metrics['avg_step_time']) / muon_metrics['avg_step_time'] * 100
    
    print(f"\nImprovement (Leo vs Muon):")
    print(f"  Validation Loss: {val_loss_improvement:+.2f}%")
    print(f"  Validation Accuracy: {accuracy_improvement:+.2f}%")
    print(f"  Step Speed: {speed_improvement:+.2f}%")
    
    # Plot results
    plot_comparison(muon_train_losses, muon_val_losses, leo_train_losses, leo_val_losses, 
                   muon_metrics, leo_metrics, config)
    
    # Save results
    results = save_results(muon_metrics, leo_metrics, config)
    
    print(f"\n🎉 Comparison completed!")
    print(f"Winner: {'Leo' if leo_metrics['final_val_loss'] < muon_metrics['final_val_loss'] else 'Muon'} "
          f"(by validation loss)")

if __name__ == "__main__":
    main()