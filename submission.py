import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math

class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 3.0))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)

class SparsityAllocator(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.routing_scores = nn.Parameter(torch.ones(num_layers))

    def forward(self, global_lambda: float):
        allocated_lambdas = F.softmax(self.routing_scores, dim=0) * len(self.routing_scores) * global_lambda
        return allocated_lambdas

class SelfPruningNetwork(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, num_classes)
        
        self.allocator = SparsityAllocator(num_layers=3)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def get_prunable_layers(self):
        return [self.fc1, self.fc2, self.fc3]

class DynamicSparsityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets, prunable_layers, layer_lambdas):
        ce = self.ce_loss(logits, targets)
        
        sparsity = torch.tensor(0.0, device=logits.device)
        for layer, lam in zip(prunable_layers, layer_lambdas):
            if lam > 0.0:
                gates = layer.get_gates()
                sparsity += lam * torch.sum(torch.abs(gates))
                
        return ce + sparsity

def calculate_sparsity_metrics(model, threshold=1e-2):
    total_elements, pruned_elements = 0, 0
    with torch.no_grad():
        for layer in model.get_prunable_layers():
            gates = layer.get_gates()
            pruned_elements += (gates < threshold).sum().item()
            total_elements += gates.numel()
    
    sparsity_ratio = (pruned_elements / total_elements) * 100 if total_elements > 0 else 0
    return total_elements, pruned_elements, sparsity_ratio

def plot_gates(model, title, filepath):
    all_gates = []
    with torch.no_grad():
        for layer in model.get_prunable_layers():
            all_gates.append(layer.get_gates().cpu().view(-1).numpy())
    
    all_gates = np.concatenate(all_gates)
    
    plt.figure(figsize=(8, 6))
    plt.hist(all_gates, bins=50, color='royalblue', alpha=0.7)
    plt.xlabel('Gate Value')
    plt.ylabel('Frequency')
    plt.title(f'Gate Values Distribution ({title})')
    plt.grid(True, alpha=0.3)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot to '{filepath}'")

def train_and_evaluate(lam_value: float, epochs: int, device: str):
    print(f"\nExperiment: Lambda = {lam_value}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=0)
    
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=256, shuffle=False, num_workers=0)

    model = SelfPruningNetwork(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = DynamicSparsityLoss()
    
    prunable_layers = model.get_prunable_layers()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits = model(x)
            layer_lambdas = model.allocator(global_lambda=lam_value)
            
            loss = criterion(logits, y, prunable_layers, layer_lambdas)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch}/{epochs} | Loss: {total_loss/len(trainloader):.4f}")

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
    _, _, sparsity_pct = calculate_sparsity_metrics(model)
    
    print(f"Test Accuracy: {val_acc:.2f}% | Sparsity: {sparsity_pct:.2f}%")
    
    return model, val_acc, sparsity_pct

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    lambda_configs = {
        "Low (1e-5)": 1e-5,
        "Medium (1e-4)": 1e-4,
        "High (1e-3)": 1e-3
    }
    
    epochs_per_run = 8
    results = []
    best_model = None
    best_sparsity = -1
    
    for name, lam in lambda_configs.items():
        model, acc, sparsity = train_and_evaluate(lam, epochs=epochs_per_run, device=device)
        results.append((name, acc, sparsity))
        
        if sparsity > best_sparsity:
            best_sparsity = sparsity
            best_model = model

    if best_model is not None:
        plot_gates(best_model, "Best Sparsity", "best_model_gate_distribution.png")

    print("\nResults Summary")
    print("-" * 50)
    print(f"{'Lambda':<15} | {'Accuracy':<15} | {'Sparsity':<15}")
    print("-" * 50)
    for name, acc, sparsity in results:
        print(f"{name:<15} | {acc:>13.2f}% | {sparsity:>13.2f}%")

if __name__ == "__main__":
    main()
