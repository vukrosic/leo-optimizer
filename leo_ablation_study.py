#!/usr/bin/env python3
"""
Leo Optimizer Ablation Study
Tests different components of Leo to understand their individual contributions
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
import seaborn as sns

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
class AblationConfig:
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
    num_documents: int = 1000
    max_tokens: int = 250000

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

# Ablation variants of Leo optimizer
class LeoAblation(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.01, 
                 align_const=0.2, eps=1e-8, 
                 # Ablation flags
                 use_row_norm=True, use_col_norm=True, use_rms_scaling=True,
                 use_lion_momentum=True, use_sign_update=False, use_adaptive_lr=False,
                 momentum_style="lion", orthog_method="element_wise"):
        """
        Leo optimizer with ablation options
        
        Args:
            use_row_norm: Whether to apply row normalization
            use_col_norm: Whether to apply column normalization  
            use_rms_scaling: Whether to apply RMS scaling
            use_lion_momentum: Whether to use Lion-style momentum vs standard momentum
            use_sign_update: Whether to use sign-based updates for 1D params
            use_adaptive_lr: Whether to use adaptive learning rate scaling
            momentum_style: "lion", "nesterov", or "standard"
            orthog_method: "element_wise", "qr", or "none"
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, align_const=align_const, eps=eps,
                       use_row_norm=use_row_norm, use_col_norm=use_col_norm, use_rms_scaling=use_rms_scaling,
                       use_lion_momentum=use_lion_momentum, use_sign_update=use_sign_update, 
                       use_adaptive_lr=use_adaptive_lr, momentum_style=momentum_style, 
                       orthog_method=orthog_method)
        super(LeoAblation, self).__init__(params, defaults)

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
                    if group['momentum_style'] == 'nesterov':
                        state['velocity'] = torch.zeros_like(p)

                momentum = state['momentum']
                
                # Different momentum styles
                if group['momentum_style'] == 'lion' and group['use_lion_momentum']:
                    # Lion-style momentum
                    update_direction = torch.lerp(grad, momentum, beta1)
                    momentum.lerp_(grad, 1 - beta2)
                elif group['momentum_style'] == 'nesterov':
                    # Nesterov momentum
                    velocity = state['velocity']
                    velocity.mul_(beta2).add_(grad, alpha=1-beta2)
                    update_direction = grad.add(velocity, alpha=beta1)
                else:
                    # Standard momentum
                    momentum.mul_(beta2).add_(grad, alpha=1-beta2)
                    update_direction = momentum

                # Orthogonalization for 2D parameters
                if p.dim() == 2 and group['orthog_method'] != 'none':
                    if group['orthog_method'] == 'element_wise':
                        # Element-wise orthogonalization (original Leo)
                        update = update_direction.clone()
                        
                        if group['use_row_norm']:
                            row_norm = torch.linalg.norm(update, ord=2, dim=1, keepdim=True)
                            update = torch.div(update, row_norm.add(eps))
                        
                        if group['use_col_norm']:
                            col_norm = torch.linalg.norm(update_direction, ord=2, dim=0, keepdim=True)
                            if group['use_row_norm']:
                                update.addcdiv_(update_direction, col_norm.add(eps))
                            else:
                                update = torch.div(update_direction, col_norm.add(eps))
                        
                        if group['use_rms_scaling']:
                            rms = torch.sqrt(torch.mean(update.square()) + eps)
                            scaling_factor = align_const / (rms + eps)
                            update.mul_(scaling_factor)
                        
                        update_direction = update
                    
                    elif group['orthog_method'] == 'qr':
                        # QR decomposition based orthogonalization
                        U, S, V = torch.svd(update_direction)
                        update_direction = U @ V.t() * align_const
                
                else:
                    # 1D parameters
                    if group['use_sign_update']:
                        update_direction = update_direction.sign_().mul_(align_const)
                    else:
                        update_direction = update_direction * align_const

                # Adaptive learning rate
                if group['use_adaptive_lr']:
                    param_norm = p.norm()
                    grad_norm = update_direction.norm()
                    if param_norm > 0 and grad_norm > 0:
                        adaptive_lr = lr * min(1.0, param_norm / (grad_norm + eps))
                    else:
                        adaptive_lr = lr
                else:
                    adaptive_lr = lr

                # Apply weight decay
                if weight_decay != 0:
                    p.add_(p, alpha=wd_factor)
                
                # Final parameter update
                p.add_(update_direction, alpha=-adaptive_lr)

        return loss

# Import model components from previous implementation
def load_and_cache_data(config: AblationConfig, cache_dir: str = "ablation_cache"):
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

        Q = self.rotary(Q.transpose(1, 2)).transpose(1, 2)
        K = self.rotary(K.transpose(1, 2)).transpose(1, 2)

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
    def __init__(self, config: AblationConfig):
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

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: AblationConfig):
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

def setup_optimizer(model: nn.Module, optimizer_config: Dict, config: AblationConfig):
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

    main_optimizer = LeoAblation(main_params, lr=config.lr, **optimizer_config)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.lr*0.1, weight_decay=config.weight_decay)

    return [main_optimizer, adamw_optimizer]

def train_ablation(ablation_name: str, optimizer_config: Dict, config: AblationConfig, 
                  train_loader: DataLoader, val_loader: DataLoader) -> Tuple[List[float], List[float], Dict]:
    """Train model with specific ablation configuration"""
    print(f"\n🧪 Training ablation: {ablation_name}")
    
    # Initialize model
    set_seed(42)  # Same seed for fair comparison
    model = MinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Total parameters: {total_params:,}")

    # Setup optimizers
    optimizers = setup_optimizer(model, optimizer_config, config)

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

    pbar = tqdm(total=config.max_steps, desc=f"Training {ablation_name}")

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
        'total_params': total_params,
        'config': optimizer_config
    }

    return train_losses, val_losses, final_metrics

def define_ablations():
    """Define all ablation configurations to test"""
    
    # Base configuration (full Leo)
    base_config = {
        'use_row_norm': True,
        'use_col_norm': True, 
        'use_rms_scaling': True,
        'use_lion_momentum': True,
        'use_sign_update': True,
        'use_adaptive_lr': False,
        'momentum_style': 'lion',
        'orthog_method': 'element_wise',
        'align_const': 0.3,
        'betas': (0.9, 0.99)
    }
    
    ablations = {
        # 1. Full Leo (baseline)
        'Leo_Full': base_config.copy(),
        
        # 2. No row normalization
        'Leo_NoRowNorm': {**base_config, 'use_row_norm': False},
        
        # 3. No column normalization  
        'Leo_NoColNorm': {**base_config, 'use_col_norm': False},
        
        # 4. No RMS scaling
        'Leo_NoRMSScaling': {**base_config, 'use_rms_scaling': False},
        
        # 5. No orthogonalization at all
        'Leo_NoOrthog': {**base_config, 'orthog_method': 'none'},
        
        # 6. Standard momentum instead of Lion
        'Leo_StandardMomentum': {**base_config, 'use_lion_momentum': False, 'momentum_style': 'standard'},
        
        # 7. Nesterov momentum
        'Leo_NesterovMomentum': {**base_config, 'momentum_style': 'nesterov'},
        
        # 8. QR decomposition orthogonalization
        'Leo_QROrthog': {**base_config, 'orthog_method': 'qr'},
        
        # 9. Different beta values (more aggressive)
        'Leo_AggressiveBetas': {**base_config, 'betas': (0.95, 0.999)},
        
        # 10. Different beta values (more conservative)
        'Leo_ConservativeBetas': {**base_config, 'betas': (0.8, 0.95)},
        
        # 11. Different alignment constant (higher)
        'Leo_HighAlign': {**base_config, 'align_const': 0.5},
        
        # 12. Different alignment constant (lower)
        'Leo_LowAlign': {**base_config, 'align_const': 0.1},
        
        # 13. With adaptive learning rate
        'Leo_AdaptiveLR': {**base_config, 'use_adaptive_lr': True},
        
        # 14. Minimal Leo (only basic orthogonalization)
        'Leo_Minimal': {
            'use_row_norm': True,
            'use_col_norm': False,
            'use_rms_scaling': False,
            'use_lion_momentum': False,
            'use_sign_update': False,
            'use_adaptive_lr': False,
            'momentum_style': 'standard',
            'orthog_method': 'element_wise',
            'align_const': 0.2,
            'betas': (0.9, 0.999)
        }
    }
    
    return ablations

def plot_ablation_results(results: Dict, config: AblationConfig):
    """Create comprehensive plots of ablation results"""
    
    # Extract metrics
    ablation_names = list(results.keys())
    val_losses = [results[name]['metrics']['final_val_loss'] for name in ablation_names]
    val_accuracies = [results[name]['metrics']['final_val_accuracy'] for name in ablation_names]
    val_perplexities = [results[name]['metrics']['final_val_perplexity'] for name in ablation_names]
    training_times = [results[name]['metrics']['training_time'] for name in ablation_names]
    step_times = [results[name]['metrics']['avg_step_time'] * 1000 for name in ablation_names]  # Convert to ms
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Validation Loss
    bars1 = ax1.bar(range(len(ablation_names)), val_losses, color='skyblue', alpha=0.7)
    ax1.set_title('Final Validation Loss by Ablation', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Ablation')
    ax1.set_ylabel('Validation Loss')
    ax1.set_xticks(range(len(ablation_names)))
    ax1.set_xticklabels(ablation_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Highlight best result
    best_idx = np.argmin(val_losses)
    bars1[best_idx].set_color('gold')
    ax1.text(best_idx, val_losses[best_idx], f'{val_losses[best_idx]:.4f}', 
             ha='center', va='bottom', fontweight='bold')
    
    # 2. Validation Accuracy
    bars2 = ax2.bar(range(len(ablation_names)), val_accuracies, color='lightgreen', alpha=0.7)
    ax2.set_title('Final Validation Accuracy by Ablation', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Ablation')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_xticks(range(len(ablation_names)))
    ax2.set_xticklabels(ablation_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Highlight best result
    best_idx = np.argmax(val_accuracies)
    bars2[best_idx].set_color('gold')
    ax2.text(best_idx, val_accuracies[best_idx], f'{val_accuracies[best_idx]:.4f}', 
             ha='center', va='bottom', fontweight='bold')
    
    # 3. Training Time
    bars3 = ax3.bar(range(len(ablation_names)), training_times, color='salmon', alpha=0.7)
    ax3.set_title('Training Time by Ablation', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Ablation')
    ax3.set_ylabel('Training Time (seconds)')
    ax3.set_xticks(range(len(ablation_names)))
    ax3.set_xticklabels(ablation_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Highlight fastest result
    best_idx = np.argmin(training_times)
    bars3[best_idx].set_color('gold')
    ax3.text(best_idx, training_times[best_idx], f'{training_times[best_idx]:.1f}s', 
             ha='center', va='bottom', fontweight='bold')
    
    # 4. Step Time
    bars4 = ax4.bar(range(len(ablation_names)), step_times, color='plum', alpha=0.7)
    ax4.set_title('Average Step Time by Ablation', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Ablation')
    ax4.set_ylabel('Step Time (ms)')
    ax4.set_xticks(range(len(ablation_names)))
    ax4.set_xticklabels(ablation_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # Highlight fastest result
    best_idx = np.argmin(step_times)
    bars4[best_idx].set_color('gold')
    ax4.text(best_idx, step_times[best_idx], f'{step_times[best_idx]:.2f}ms', 
             ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'leo_ablation_results_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Ablation results plot saved as {filename}")
    
    plt.show()
    
    # Create training curves plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot training loss curves for top 5 ablations
    sorted_ablations = sorted(results.items(), key=lambda x: x[1]['metrics']['final_val_loss'])[:5]
    
    for name, data in sorted_ablations:
        steps = list(range(0, len(data['val_losses']) * config.eval_every, config.eval_every))
        ax1.plot(steps, data['val_losses'], label=name, marker='o', markersize=3)
    
    ax1.set_title('Validation Loss Curves (Top 5 Ablations)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot training loss curves (smoothed)
    for name, data in sorted_ablations:
        # Smooth training losses
        window = 50
        if len(data['train_losses']) > window:
            smoothed = np.convolve(data['train_losses'], np.ones(window)/window, mode='valid')
            ax2.plot(smoothed, label=name, alpha=0.8)
    
    ax2.set_title('Smoothed Training Loss Curves (Top 5 Ablations)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Training Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save training curves
    filename = f'leo_ablation_curves_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Training curves plot saved as {filename}")
    
    plt.show()

def save_ablation_results(results: Dict, config: AblationConfig):
    """Save detailed ablation results to JSON"""
    
    # Prepare results for JSON serialization
    json_results = {}
    for name, data in results.items():
        json_results[name] = {
            'metrics': data['metrics'],
            'config_summary': {
                'use_row_norm': data['metrics']['config']['use_row_norm'],
                'use_col_norm': data['metrics']['config']['use_col_norm'],
                'use_rms_scaling': data['metrics']['config']['use_rms_scaling'],
                'use_lion_momentum': data['metrics']['config']['use_lion_momentum'],
                'momentum_style': data['metrics']['config']['momentum_style'],
                'orthog_method': data['metrics']['config']['orthog_method'],
                'align_const': data['metrics']['config']['align_const'],
                'betas': data['metrics']['config']['betas']
            }
        }
    
    # Add experiment metadata
    experiment_data = {
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
        'ablations': json_results,
        'summary': {
            'best_val_loss': min(data['metrics']['final_val_loss'] for data in results.values()),
            'best_accuracy': max(data['metrics']['final_val_accuracy'] for data in results.values()),
            'fastest_training': min(data['metrics']['training_time'] for data in results.values()),
            'fastest_step': min(data['metrics']['avg_step_time'] for data in results.values())
        }
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'leo_ablation_results_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"📄 Detailed results saved to {filename}")
    return experiment_data

def print_ablation_summary(results: Dict):
    """Print a summary of ablation results"""
    print("\n" + "="*80)
    print("🧪 LEO OPTIMIZER ABLATION STUDY RESULTS")
    print("="*80)
    
    # Sort by validation loss
    sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['final_val_loss'])
    
    print(f"\n{'Rank':<4} {'Ablation':<20} {'Val Loss':<10} {'Val Acc':<10} {'Time(s)':<10} {'Step(ms)':<10}")
    print("-" * 80)
    
    for i, (name, data) in enumerate(sorted_results, 1):
        metrics = data['metrics']
        print(f"{i:<4} {name:<20} {metrics['final_val_loss']:<10.4f} "
              f"{metrics['final_val_accuracy']:<10.4f} {metrics['training_time']:<10.1f} "
              f"{metrics['avg_step_time']*1000:<10.2f}")
    
    # Key insights
    print(f"\n🏆 KEY INSIGHTS:")
    best_loss = sorted_results[0]
    best_acc = max(results.items(), key=lambda x: x[1]['metrics']['final_val_accuracy'])
    fastest = min(results.items(), key=lambda x: x[1]['metrics']['training_time'])
    
    print(f"   Best Validation Loss: {best_loss[0]} ({best_loss[1]['metrics']['final_val_loss']:.4f})")
    print(f"   Best Accuracy: {best_acc[0]} ({best_acc[1]['metrics']['final_val_accuracy']:.4f})")
    print(f"   Fastest Training: {fastest[0]} ({fastest[1]['metrics']['training_time']:.1f}s)")
    
    # Component analysis
    print(f"\n🔍 COMPONENT ANALYSIS:")
    full_leo = results['Leo_Full']['metrics']['final_val_loss']
    
    components = [
        ('Row Normalization', 'Leo_NoRowNorm'),
        ('Column Normalization', 'Leo_NoColNorm'), 
        ('RMS Scaling', 'Leo_NoRMSScaling'),
        ('Orthogonalization', 'Leo_NoOrthog'),
        ('Lion Momentum', 'Leo_StandardMomentum')
    ]
    
    for component, ablation_name in components:
        if ablation_name in results:
            ablation_loss = results[ablation_name]['metrics']['final_val_loss']
            impact = ((ablation_loss - full_leo) / full_leo) * 100
            print(f"   {component}: {impact:+.2f}% impact when removed")

def main():
    """Main ablation study function"""
    print("🧪 Leo Optimizer Ablation Study")
    print("=" * 60)
    
    # Check system
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create config
    config = AblationConfig()
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
    
    # Define ablations
    ablations = define_ablations()
    print(f"\n🧪 Running {len(ablations)} ablations:")
    for name in ablations.keys():
        print(f"   - {name}")
    
    # Run ablations
    results = {}
    total_start_time = time.time()
    
    for i, (ablation_name, optimizer_config) in enumerate(ablations.items(), 1):
        print(f"\n{'='*60}")
        print(f"Running ablation {i}/{len(ablations)}: {ablation_name}")
        print(f"{'='*60}")
        
        train_losses, val_losses, metrics = train_ablation(
            ablation_name, optimizer_config, config, train_loader, val_loader
        )
        
        results[ablation_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'metrics': metrics
        }
    
    total_time = time.time() - total_start_time
    print(f"\n🎉 All ablations completed in {total_time/60:.1f} minutes!")
    
    # Analyze and save results
    print_ablation_summary(results)
    plot_ablation_results(results, config)
    save_ablation_results(results, config)

if __name__ == "__main__":
    main()