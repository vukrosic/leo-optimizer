import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Optimizer
import time
from datetime import datetime, timedelta

torch.set_num_threads(8)

# ------------------- GPU设备检测 -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ------------------- 定义Leo优化器 (保持不变) -------------------
import math  # Add this import at the top of your file if not already present

class Leo(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.01, align_const=0.2, eps=1e-8):
        """
        Initializes the Leo (Lion with Element-wise Orthogonalization-proxy) optimizer.
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients for momentum update (default: (0.9, 0.99))
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0.01)
            align_const (float, optional): The 'C' constant for RMS alignment, from Muon. (default: 0.2)
            eps (float, optional): A small epsilon for numerical stability in normalization. (default: 1e-8)
        """
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
            # OPTIMIZATION: Pre-calculate constants outside the parameter loop
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            align_const = group['align_const']
            eps = group['eps']
            
            # OPTIMIZATION: Pre-calculate weight decay factor if it's used
            # This avoids repeated multiplication inside the loop.
            # p.add_(p, alpha) is often preferred over p.mul_ for weight decay.
            wd_factor = -lr * weight_decay

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)

                momentum = state['momentum']
                
                # The update direction is calculated using the *previous* momentum state
                # This is a non-trivial detail from the original code that we must preserve.
                # update_direction = beta1 * m_{t-1} + (1-beta1) * g_t
                update_direction = torch.lerp(grad, momentum, beta1)

                # The momentum is then updated for the *next* step (Lion-style)
                # m_t = beta2 * m_{t-1} + (1-beta2) * g_t
                momentum.lerp_(grad, 1 - beta2)

                # --- Core logic for update direction processing ---
                if p.dim() == 2:
                    # OPTIMIZATION: Use the modern and recommended linalg.norm
                    row_norm = torch.linalg.norm(update_direction, ord=2, dim=1, keepdim=True)
                    col_norm = torch.linalg.norm(update_direction, ord=2, dim=0, keepdim=True)
                    
                    # The original use of `div` and `addcdiv_` is already quite efficient
                    # as `addcdiv` is a fused kernel. We keep this structure.
                    # update = update_direction / (row_norm + eps) + update_direction / (col_norm + eps)
                    update = torch.div(update_direction, row_norm.add(eps))
                    update.addcdiv_(update_direction, col_norm.add(eps))
                    
                    # OPTIMIZATION: More direct and numerically stable RMS calculation.
                    # RMS = sqrt(mean(x^2)). Add eps inside sqrt for stability when update is all zeros.
                    rms = torch.sqrt(torch.mean(update.square()) + eps)
                    
                    # OPTIMIZATION: Fuse multiplication and division into a single operation
                    # by pre-calculating the scaling factor.
                    scaling_factor = align_const / (rms + eps)
                    update.mul_(scaling_factor)
                    
                    update_direction = update
                else:
                    # This branch remains the same but can be chained for style.
                    update_direction.sign_().mul_(align_const)

                # Apply weight decay (if any)
                if weight_decay != 0:
                    p.add_(p, alpha=wd_factor)
                
                # Final parameter update
                p.add_(update_direction, alpha=-lr)

        return loss

# ------------------- 定义适用于CIFAR-10的CNN模型 -------------------
# MODIFIED: The CNN architecture is updated for CIFAR-10's 3x32x32 images.
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # MODIFIED: Input channels changed from 1 (MNIST) to 3 (CIFAR-10)
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1) # Added padding to maintain size
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2) # 32x32 -> 16x16 -> 8x8
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # MODIFIED: Recalculate the input features for the fully connected layer.
        # Input: 32x32
        # After conv1 (with padding=1): 32x32
        # After pool1: 16x16
        # After conv2 (with padding=1): 16x16
        # After pool2: 8x8
        # Flattened size: 64 channels * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 128) 
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x) # First pool
        x = self.relu(self.conv2(x))
        x = self.pool(x) # Second pool
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# ------------------- 准备CIFAR-10数据 -------------------
# MODIFIED: Changed normalization for CIFAR-10 (3 channels)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizing to [-1, 1] range
])

# MODIFIED: Using torchvision.datasets.CIFAR10 instead of MNIST
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2048*4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# NEW: Define class names for context (optional, but good practice)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ------------------- 训练函数 (增强日志) -------------------
def train_model(optimizer_class, optimizer_name, lr=0.001, weight_decay=0.01, num_epochs=10, **kwargs):
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    
    train_losses = []
    
    # Training setup info
    total_batches = len(train_loader)
    total_steps = num_epochs * total_batches
    samples_per_batch = train_loader.batch_size
    total_samples = len(train_dataset)
    
    print(f"\n{'='*60}")
    print(f"Starting Training: {optimizer_name}")
    print(f"{'='*60}")
    print(f"Model: SimpleCNN | Dataset: CIFAR-10")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs} | Batch Size: {samples_per_batch}")
    print(f"Total Batches per Epoch: {total_batches}")
    print(f"Total Training Steps: {total_steps}")
    print(f"Learning Rate: {lr} | Weight Decay: {weight_decay}")
    if kwargs:
        print(f"Optimizer Params: {kwargs}")
    print(f"{'='*60}")
    
    # Global timing
    global_start_time = time.time()
    step_count = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        epoch_samples = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"{'-'*50}")
        
        for i, (inputs, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            batch_time = time.time() - batch_start_time
            running_loss += loss.item()
            step_count += 1
            epoch_samples += inputs.size(0)
            
            # Detailed logging every 5 batches for better tracking
            if i % 5 == 4:
                avg_loss = running_loss / 5
                
                # Speed calculations
                elapsed_time = time.time() - global_start_time
                steps_per_sec = step_count / elapsed_time
                samples_per_sec = (epoch * total_samples + epoch_samples) / elapsed_time
                
                # ETA calculations
                remaining_steps = total_steps - step_count
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                eta_time = datetime.now() + timedelta(seconds=eta_seconds)
                
                # Progress percentage
                progress = (step_count / total_steps) * 100
                
                print(f"Step {step_count:4d}/{total_steps} [{progress:5.1f}%] | "
                      f"Batch {i+1:3d}/{total_batches} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Speed: {steps_per_sec:.1f} steps/s, {samples_per_sec:.0f} samples/s | "
                      f"ETA: {eta_time.strftime('%H:%M:%S')}")
                
                running_loss = 0.0
        
        # Epoch evaluation
        model.eval()
        total_loss = 0.0
        eval_start_time = time.time()
        
        with torch.no_grad():
            for inputs, labels in test_loader:  # Use test_loader for evaluation
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        
        eval_time = time.time() - eval_start_time
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(test_loader)
        train_losses.append(avg_loss)
        
        # Epoch summary
        elapsed_total = time.time() - global_start_time
        remaining_epochs = num_epochs - (epoch + 1)
        avg_epoch_time = elapsed_total / (epoch + 1)
        eta_total = remaining_epochs * avg_epoch_time
        eta_finish = datetime.now() + timedelta(seconds=eta_total)
        
        print(f"\nEpoch {epoch+1} Complete:")
        print(f"  Training Time: {epoch_time:.1f}s | Eval Time: {eval_time:.1f}s")
        print(f"  Test Loss: {avg_loss:.4f}")
        print(f"  Avg Epoch Time: {avg_epoch_time:.1f}s | ETA Finish: {eta_finish.strftime('%H:%M:%S')}")
        print(f"  Total Elapsed: {elapsed_total/60:.1f}m")
    
    # Final summary
    total_time = time.time() - global_start_time
    final_steps_per_sec = total_steps / total_time
    final_samples_per_sec = (num_epochs * total_samples) / total_time
    
    print(f"\n{'='*60}")
    print(f"Training Complete: {optimizer_name}")
    print(f"{'='*60}")
    print(f"Total Time: {total_time/60:.1f} minutes ({total_time:.1f}s)")
    print(f"Final Speed: {final_steps_per_sec:.1f} steps/s, {final_samples_per_sec:.0f} samples/s")
    print(f"Final Test Loss: {train_losses[-1]:.4f}")
    print(f"{'='*60}")
    
    return train_losses

# ------------------- 更新实验设置 -------------------
# 设置超参数
lr = 3e-4
weight_decay = 0.01
num_epochs = 35 # CIFAR-10 is more complex, you might need more epochs for good results

# Leo specific hyperparameters
leo_params = {
    'betas': (0.9, 0.99),
    'align_const': 0.3
}

# 训练并比较两种优化器
leo_losses = train_model(Leo, "Leo",lr=lr, weight_decay=weight_decay, num_epochs=num_epochs, **leo_params)
adamw_losses = train_model(optim.AdamW, "AdamW",lr=3e-4, weight_decay=weight_decay, num_epochs=num_epochs)

# ------------------- 绘制Loss曲线图 -------------------
print(f"\n{'='*60}")
print("Generating Training Comparison Plot...")
print(f"{'='*60}")

plt.figure(figsize=(12, 7))
plt.plot(range(1, num_epochs+1), leo_losses, label='Leo', marker='o', linewidth=2, markersize=6, color='#1f77b4')
plt.plot(range(1, num_epochs+1), adamw_losses, label='AdamW', marker='s', linewidth=2, markersize=6, color='#ff7f0e')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Test Loss', fontsize=12)
plt.title('Comparison of Leo vs AdamW Optimizers on CIFAR-10\nTraining Loss Over Time', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
filename = 'optimizer_comparison_leo_cifar10.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Plot saved as: {filename}")

# Display the plot
print("Displaying comparison plot... (Close the plot window to continue)")
plt.show(block=True)

# Print final results
print(f"\n{'='*60}")
print("EXPERIMENT COMPLETE")
print(f"{'='*60}")
print(f"\nFinal Test Loss Results:")
print(f"Leo Final Loss: {leo_losses[-1]:.4f}")
print(f"AdamW Final Loss: {adamw_losses[-1]:.4f}")
print(f"Leo performed {'better' if leo_losses[-1] < adamw_losses[-1] else 'worse'} than AdamW")
print(f"{'='*60}")