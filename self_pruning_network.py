"""
Self-Pruning Neural Network with Straight-Through Hard Gates
============================================================
Tredence Analytics — AI Engineering Intern Case Study
Author: Supriya Pallisetty

Unique Approach:
    Standard solutions use soft sigmoid gates that approach-but-never-reach 0.
    This implementation uses a Straight-Through Estimator (STE) to allow truly
    binary (0 or 1) gates during the forward pass, while still allowing gradients
    to flow during the backward pass. This produces cleaner, more interpretable
    pruning compared to soft-gate alternatives.

Architecture:
    PrunableLinear  →  custom layer with learnable gate_scores
    SelfPruningNet  →  3-layer feed-forward network for CIFAR-10
    Training loop   →  Total Loss = CrossEntropy + λ * L1(gates)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — PrunableLinear Layer
# ─────────────────────────────────────────────────────────────────────────────

class StraightThroughStep(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for hard binarization.

    Forward:  gate = 1  if sigmoid(score) > 0.5,  else 0   (hard / binary)
    Backward: gradient passes through as if the step didn't exist (identity)

    Why this matters:
        Soft sigmoid gates (the naive approach) never truly reach 0; they
        asymptotically approach it. This means a "pruned" network still carries
        small non-zero weights that consume memory. STE lets us use real binary
        gates during inference while training with gradient descent — a technique
        from Bengio et al. (2013) widely used in quantized neural networks.
    """
    @staticmethod
    def forward(ctx, scores):
        # Hard threshold at 0.5 in sigmoid space (i.e., score > 0)
        return (torch.sigmoid(scores) > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: pass gradient unchanged
        return grad_output


class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that learns which weights to prune.

    Each weight w_ij has an associated learnable gate_score g_ij.
    During the forward pass:
        gates       = STE_binarize(sigmoid(gate_scores))   # binary: 0 or 1
        soft_gates  = sigmoid(gate_scores)                 # used only for L1 loss
        pruned_w    = weight * gates
        output      = pruned_w @ input.T + bias

    The soft_gates are exposed via self.soft_gates so the training loop
    can compute the L1 sparsity penalty on continuous values (necessary for
    gradients to flow into gate_scores).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias — same as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Gate scores — same shape as weight; initialized near 0.5 in sigmoid space
        # Initializing to small positive values gives gates a slight open bias
        # so the network starts dense and learns to prune, not the reverse
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        # Internal cache for sparsity loss computation
        self.soft_gates = None

        self._init_parameters()

    def _init_parameters(self):
        # Kaiming uniform for weights (standard for ReLU networks)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        # Gate scores near 0 → sigmoid(0) = 0.5, so ~50% gates open initially
        nn.init.normal_(self.gate_scores, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute soft gates (continuous, for L1 penalty)
        self.soft_gates = torch.sigmoid(self.gate_scores)

        # Compute hard binary gates using STE (for actual pruning)
        hard_gates = StraightThroughStep.apply(self.gate_scores)

        # Element-wise multiply: weights that are gated off become exactly 0
        pruned_weights = self.weight * hard_gates

        # Standard linear transformation
        return F.linear(x, pruned_weights, self.bias)

    def sparsity(self, threshold: float = 1e-2) -> float:
        """Return fraction of weights whose gate is below threshold."""
        if self.soft_gates is None:
            with torch.no_grad():
                self.soft_gates = torch.sigmoid(self.gate_scores)
        below = (self.soft_gates < threshold).float().sum()
        total = self.soft_gates.numel()
        return (below / total).item()

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 (cont.) — Network Definition
# ─────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    3-layer feed-forward network for CIFAR-10 using PrunableLinear layers.

    Input:  32×32×3 images  →  flattened to 3072-dim vector
    Hidden: 1024 → 512 → 256
    Output: 10 classes

    BatchNorm is added after each hidden layer for training stability.
    Dropout is intentionally omitted — the gating mechanism IS the regularizer.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        self.fc1 = PrunableLinear(3072, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

    def prunable_layers(self):
        """Yield all PrunableLinear layers in the network."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all soft gate values across every PrunableLinear layer.

        Why L1 (not L2)?
            L2 penalizes large values heavily but lets small values persist.
            L1 applies a constant downward pressure on ALL values, which drives
            them to exactly zero — a property called "sparsity inducing."
            This is the same reason LASSO regression produces sparse solutions.
        """
        total = torch.tensor(0.0, requires_grad=True)
        for layer in self.prunable_layers():
            if layer.soft_gates is not None:
                total = total + layer.soft_gates.sum()
        return total

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of all prunable weights that are effectively zeroed out."""
        total_weights, pruned_weights = 0, 0
        for layer in self.prunable_layers():
            _ = layer.forward  # ensure soft_gates populated
            with torch.no_grad():
                sg = torch.sigmoid(layer.gate_scores)
            pruned_weights += (sg < threshold).sum().item()
            total_weights  += sg.numel()
        return pruned_weights / total_weights if total_weights > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 128):
    """Load CIFAR-10 with standard augmentation for training."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=train_transform)
    test_ds  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Training & Evaluation Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, lambda_sparse, device):
    model.train()
    total_loss = total_cls_loss = total_sparse_loss = 0.0
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(images)

        # Classification loss
        cls_loss = F.cross_entropy(logits, labels)

        # Sparsity regularization loss (L1 on sigmoid gates)
        sparse_loss = model.sparsity_loss()

        # Total loss: trade-off controlled by lambda
        loss = cls_loss + lambda_sparse * sparse_loss

        loss.backward()

        # Gradient clipping prevents exploding gradients in gate_scores
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss        += loss.item()
        total_cls_loss    += cls_loss.item()
        total_sparse_loss += sparse_loss.item()
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    n = len(loader)
    return {
        "loss":         total_loss / n,
        "cls_loss":     total_cls_loss / n,
        "sparse_loss":  total_sparse_loss / n,
        "accuracy":     correct / total,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


def train_model(lambda_sparse: float, epochs: int = 25, seed: int = 42):
    """
    Full training run for a given lambda value.
    Returns: (test_accuracy, sparsity_level, gate_values_flat)
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_cifar10_loaders()
    model = SelfPruningNet().to(device)

    # Adam with weight decay on parameters — but NOT on gate_scores
    # We separate params so weight decay doesn't interfere with gate training
    gate_params   = [p for n, p in model.named_parameters() if "gate_scores" in n]
    weight_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]

    optimizer = torch.optim.Adam([
        {"params": weight_params, "weight_decay": 1e-4},
        {"params": gate_params,   "weight_decay": 0.0},
    ], lr=1e-3)

    # Cosine annealing gradually reduces LR for stable convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}")
    print(f"  Training with λ = {lambda_sparse}")
    print(f"  Device: {device} | Epochs: {epochs}")
    print(f"{'='*60}")

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        stats = train_epoch(model, train_loader, optimizer, lambda_sparse, device)
        val_acc = evaluate(model, test_loader, device)
        scheduler.step()

        sparsity = model.overall_sparsity()

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"Loss: {stats['loss']:.4f}  "
                  f"Cls: {stats['cls_loss']:.4f}  "
                  f"Sparse: {stats['sparse_loss']:.1f}  "
                  f"Val Acc: {val_acc*100:.1f}%  "
                  f"Sparsity: {sparsity*100:.1f}%  "
                  f"({time.time()-t0:.1f}s)")

    # Collect all gate values for histogram
    gate_values = []
    for layer in model.prunable_layers():
        with torch.no_grad():
            sg = torch.sigmoid(layer.gate_scores).cpu().numpy().flatten()
            gate_values.append(sg)
    gate_values_flat = np.concatenate(gate_values)

    final_test_acc = evaluate(model, test_loader, device)
    final_sparsity = model.overall_sparsity()

    print(f"\n  ✓ Final Test Accuracy : {final_test_acc*100:.2f}%")
    print(f"  ✓ Final Sparsity      : {final_sparsity*100:.2f}%")

    return final_test_acc, final_sparsity, gate_values_flat


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — Visualization & Report
# ─────────────────────────────────────────────────────────────────────────────

def plot_gate_distributions(results: dict, save_path: str = "gate_distributions.png"):
    """
    Plot gate value histograms for each lambda.
    A successful pruning shows a bimodal distribution: spike at 0, cluster near 1.
    """
    lambdas = sorted(results.keys())
    fig, axes = plt.subplots(1, len(lambdas), figsize=(5 * len(lambdas), 4))

    colors = ["#2196F3", "#FF9800", "#F44336"]

    for ax, lam, color in zip(axes, lambdas, colors):
        gates = results[lam]["gates"]
        acc   = results[lam]["accuracy"] * 100
        spar  = results[lam]["sparsity"] * 100

        ax.hist(gates, bins=60, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_title(f"λ = {lam}\nAcc: {acc:.1f}%  |  Sparsity: {spar:.1f}%",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Gate Value (sigmoid output)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_xlim(0, 1)
        ax.axvline(0.01, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="Prune threshold (0.01)")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Gate Value Distributions — Self-Pruning Network (STE Hard Gates)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Plot saved → {save_path}")


def print_summary_table(results: dict):
    """Print a markdown-compatible results table."""
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Lambda':<12} {'Test Accuracy':>14} {'Sparsity (%)':>14}")
    print(f"  {'-'*12} {'-'*14} {'-'*14}")
    for lam in sorted(results.keys()):
        acc  = results[lam]["accuracy"] * 100
        spar = results[lam]["sparsity"] * 100
        print(f"  {lam:<12} {acc:>13.2f}% {spar:>13.2f}%")
    print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Three lambda values: low / medium / high sparsity pressure
    LAMBDAS = [1e-5, 1e-4, 5e-4]
    EPOCHS  = 25   # Increase to 50+ for production-quality results

    results = {}
    best_gates = None
    best_lam   = None

    for lam in LAMBDAS:
        acc, sparsity, gates = train_model(lambda_sparse=lam, epochs=EPOCHS)
        results[lam] = {
            "accuracy": acc,
            "sparsity": sparsity,
            "gates":    gates,
        }
        # Track the model with highest accuracy for the plot
        if best_gates is None or acc > results.get(best_lam, {}).get("accuracy", 0):
            best_gates = gates
            best_lam   = lam

    print_summary_table(results)
    plot_gate_distributions(results, save_path="gate_distributions.png")

    print("\n  All done. Submit gate_distributions.png along with this script.")
