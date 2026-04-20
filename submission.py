"""
Tredence Analytics AI Engineering Assignment
Case Study: The Self-Pruning Neural Network

This script provides an end-to-end implementation of a neural network that learns
to dynamically prune its own weights during training. It employs a custom PrunableLinear
layer featuring learnable gating mechanisms.

Key Feature (Bonus): Learnable Sparsity Budget Allocation
Rather than applying a uniform penalty across all layers, this implementation includes
a 'SparsityAllocator' that dynamically routes the L1 penalty to different layers based
on a learnable softmax mechanism. This pushes pruning to layers where it hurts the least!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math

# ==========================================================
# PART 1: The "Prunable" Layers & Architecture
# ==========================================================

class PrunableLinear(nn.Module):
    """
    A custom linear layer where weights are multiplied by learnable sigmoid gates.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # The learnable gate parameters
        # Initializing them to a high value (like 3.0) means sigmoid(3) ~ 0.95.
        # This allows the network to start training with mostly "open" gates, 
        # finding good features before aggressive pruning begins.
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 3.0))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Kaiming uniform initialization for stability."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying the dynamic gating mechanism.
        """
        # Transform unbounded gate scores to 0-1 range
        gates = torch.sigmoid(self.gate_scores)
        # Element-wise multiplication to functionally prune weights
        pruned_weights = self.weight * gates
        # Standard linear projection with the now-sparsified weights
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Returns the activated gate values for sparsity tracking & L1 penalization."""
        return torch.sigmoid(self.gate_scores)

class SparsityAllocator(nn.Module):
    """
    [BONUS/UNIQUE FEATURE] Dynamically shifts penalization weight between layers 
    so the network can computationally decide which layers to prune heavily 
    and which delicate layers to keep largely intact.
    """
    def __init__(self, num_layers: int):
        super().__init__()
        # Initialize routing scores equally.
        self.routing_scores = nn.Parameter(torch.ones(num_layers))

    def forward(self, global_lambda: float):
        # Softmax ensures total penalty sum remains constant despite re-routing.
        allocated_lambdas = F.softmax(self.routing_scores, dim=0) * len(self.routing_scores) * global_lambda
        return allocated_lambdas

class SelfPruningNetwork(nn.Module):
    """
    A standard Multilayer Perceptron (MLP) for CIFAR-10 leveraging Prunable layers.
    (JD requested a feed-forward neural network for classification)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Flatten CIFAR-10 images (3 channels * 32 * 32 = 3072)
        self.flatten = nn.Flatten()
        
        # A 3-layer fully connected network using our PrunableLinear implementation
        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, num_classes)
        
        # Manages lambda distribution among the 3 prunable layers
        self.allocator = SparsityAllocator(num_layers=3)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def get_prunable_layers(self):
        """Returns references to all prunable layers for metric/loss evaluations."""
        return [self.fc1, self.fc2, self.fc3]

# ==========================================================
# PART 2: The Sparsity Regularization Loss
# ==========================================================

class DynamicSparsityLoss(nn.Module):
    """
    Calculates Classification Loss (Cross Entropy) + L1 penalty on the gate values.
    """
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets, prunable_layers, layer_lambdas):
        # 1. Base classification loss
        ce = self.ce_loss(logits, targets)
        
        # 2. L1 Sparsity Loss computed across all gates
        sparsity = torch.tensor(0.0, device=logits.device)
        for layer, lam in zip(prunable_layers, layer_lambdas):
            if lam > 0.0:
                gates = layer.get_gates()
                # Sum of absolute values (gates are strictly positive 0-1 due to sigmoid)
                sparsity += lam * torch.sum(torch.abs(gates))
                
        # Total Loss = ClassificationLoss + lambda * SparsityLoss
        return ce + sparsity


# ==========================================================
# PART 3: Training and Evaluation Loops
# ==========================================================

def calculate_sparsity_metrics(model, threshold=1e-2):
    """
    Calculates the exact percentage of weights effectively pruned (< threshold).
    """
    total_elements, pruned_elements = 0, 0
    with torch.no_grad():
        for layer in model.get_prunable_layers():
            gates = layer.get_gates()
            pruned_elements += (gates < threshold).sum().item()
            total_elements += gates.numel()
    
    sparsity_ratio = (pruned_elements / total_elements) * 100 if total_elements > 0 else 0
    return total_elements, pruned_elements, sparsity_ratio

def plot_gates(model, title, filepath):
    """Generates the required Matplotlib plot showing gate value distributions."""
    all_gates = []
    with torch.no_grad():
        for layer in model.get_prunable_layers():
            all_gates.append(layer.get_gates().cpu().view(-1).numpy())
    
    all_gates = np.concatenate(all_gates)
    
    plt.figure(figsize=(8, 6))
    plt.hist(all_gates, bins=50, color='royalblue', alpha=0.7)
    plt.xlabel('Gate Value (0.0 to 1.0)')
    plt.ylabel('Frequency (Count of Parameters)')
    plt.title(f'Learned Gate Values Distribution ({title})')
    plt.grid(True, alpha=0.3)
    plt.savefig(filepath)
    plt.close()
    print(f"[*] Distribution plot successfully saved to '{filepath}'")

def train_and_evaluate(lam_value: float, epochs: int, device: str):
    """
    Executes a complete training and evaluation run for a specific global lambda.
    """
    print(f"\n{'='*50}\nExperiment: Global Lambda = {lam_value}\n{'='*50}")
    
    # 1. Dataset Prep
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=0)
    
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=256, shuffle=False, num_workers=0)

    # 2. Model & Optimizer Setup
    model = SelfPruningNetwork(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = DynamicSparsityLoss()
    
    prunable_layers = model.get_prunable_layers()

    # 3. Epoch Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits = model(x)
            
            # The Allocator dynamically routes the global lambda to the layers
            layer_lambdas = model.allocator(global_lambda=lam_value)
            
            loss = criterion(logits, y, prunable_layers, layer_lambdas)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch}/{epochs} | Average Total Loss: {total_loss/len(trainloader):.4f}")

    # 4. Final Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in valloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
    val_acc = 100.0 * correct / total
    _, _, sparsity_pct = calculate_sparsity_metrics(model, threshold=1e-2)
    
    print(f"--> Final Test Accuracy: {val_acc:.2f}%")
    print(f"--> Final Sparsity Level: {sparsity_pct:.2f}%")
    
    return model, val_acc, sparsity_pct

def main():
    """Main execution entry point to run multiple experiments back-to-back."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    
    # JD Requirement: Compare results for at least 3 different values of lambda
    lambda_configs = {
        "Low (0.00001)": 0.00001,
        "Medium (0.0001)": 0.0001,
        "High (0.001)": 0.001
    }
    
    # Using 8 epochs for time-efficiency in demonstration, scale up in production!
    epochs_per_run = 8
    
    results = []
    best_model = None
    best_sparsity = -1
    
    for name, lam in lambda_configs.items():
        model, acc, sparsity = train_and_evaluate(lam, epochs=epochs_per_run, device=device)
        results.append((name, acc, sparsity))
        
        # Save structural details for the model that pruned the most efficiently
        if sparsity > best_sparsity:
            best_sparsity = sparsity
            best_model = model

    # Generate the requested plot
    if best_model is not None:
        plot_gates(best_model, "Highest Sparsity Model", "best_model_gate_distribution.png")

    # Generate the Markdown Table in terminal
    print("\n" + "="*53)
    print("FINAL RESULTS TABLE")
    print("="*53)
    print(f"{'Lambda Config':<15} | {'Test Accuracy':<15} | {'Sparsity Level':<15}")
    print("-" * 53)
    for name, acc, sparsity in results:
        print(f"{name:<15} | {acc:>13.2f}% | {sparsity:>13.2f}%")

if __name__ == "__main__":
    main()
