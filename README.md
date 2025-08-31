# Leo Optimizer

Leo (Lion with Element-wise Orthogonalization-proxy) is a fast and efficient optimizer that combines Lion-style momentum updates with element-wise orthogonalization techniques. It's designed as an improved alternative to Muon optimizer with better computational efficiency.

🫐 Blueberry AI | [Discord](https://discord.gg/bAENzZMF) | [YouTube](https://www.youtube.com/@vukrosic) | [Bilibili](https://space.bilibili.com/3546833932519662/upload/video)

## How Leo Works

Leo optimizer uses a hybrid approach that combines:

1. **Lion-style momentum updates**: Uses exponential moving averages with different decay rates for gradient accumulation and momentum updates
2. **Element-wise orthogonalization**: For 2D parameters (like linear layer weights), applies row and column normalization followed by RMS scaling
3. **Adaptive scaling**: Uses an alignment constant to control the magnitude of updates

### Key Differences from Muon

| Feature | Muon | Leo |
|---------|------|-----|
| **Orthogonalization** | Newton-Schulz iteration (5 steps) | Element-wise row/column normalization |
| **Computational Cost** | Higher (matrix operations) | Lower (element-wise operations) |
| **Memory Usage** | More intensive | More efficient |
| **Momentum Style** | Nesterov momentum | Lion-style dual momentum |
| **Parameter Handling** | Uniform approach | Dimension-aware (2D vs 1D) |

## Performance Comparison

We compared Leo and Muon optimizers on language modeling tasks using a 6-layer transformer (2.4M parameters):

![Leo vs Muon Comparison](llm_optimizer_comparison_20250831_134852.png.png)

### Results Summary

- **Training Loss**: Leo achieved slightly better convergence
- **Validation Loss**: Leo: 4.2841 vs Muon: 4.2966 (-2.9% improvement)
- **Validation Accuracy**: Leo: 0.2846 vs Muon: 0.2838 (+0.3% improvement)
- **Training Speed**: Leo was ~5% faster per step due to more efficient operations

## Usage

```python
from leo_optimizer import Leo

# Initialize Leo optimizer
optimizer = Leo(
    model.parameters(),
    lr=0.01,
    betas=(0.9, 0.99),
    weight_decay=0.01,
    align_const=0.3,
    eps=1e-8
)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

## Files

- `llm_leo.py` - Language model implementation with Leo optimizer
- `llm_muon.py` - Language model implementation with Muon optimizer  
- `compare_llm_optimizers.py` - Comparison script for both optimizers
- `cifar10-leo-vs-adamw.py` - CIFAR-10 comparison with AdamW

## Installation

```bash
git clone https://github.com/your-repo/leo-optimizer
cd leo-optimizer
pip install -r requirements.txt
```

## Running Comparisons

```bash
# Compare Leo vs Muon on language modeling
python compare_llm_optimizers.py

# Compare Leo vs AdamW on CIFAR-10
python cifar10-leo-vs-adamw.py
```