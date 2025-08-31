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


# ------------------- 训练函数 (保持不变) -------------------
def train_model(optimizer_class, optimizer_name, lr=0.001, weight_decay=0.01, num_epochs=10, **kwargs):
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    
    train_losses = []
    
    print(f"\n--- Starting Training for {optimizer_name} ---")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print more frequently for the larger CIFAR-10 dataset
            if i % 200 == 199:
                print(f'{optimizer_name} - Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/200:.4f}')
                running_loss = 0.0
        
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'{optimizer_name} - Epoch {epoch+1} Finished, Average Training Loss: {avg_loss:.4f}')
    
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

# ------------------- 绘制Loss曲线图 (保持不变) -------------------
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), leo_losses, label='Leo', marker='o')
plt.plot(range(1, num_epochs+1), adamw_losses, label='AdamW', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Comparison of Leo and AdamW Optimizers on CIFAR-10') # MODIFIED: Updated title
plt.legend()
plt.grid(True)
plt.savefig('optimizer_comparison_leo_cifar10.png') # MODIFIED: Updated filename
plt.show()

# 打印最终结果
print(f"\nFinal Average Training Loss:")
print(f"Leo Final Loss: {leo_losses[-1]:.4f}")
print(f"AdamW Final Loss: {adamw_losses[-1]:.4f}")