"""
Mean-Shifted Contrastive (MSC) Loss for FENCE.

Pulls positive samples into a tight cluster while pushing negatives away.
Based on: Reiss & Hoshen, "Mean-Shifted Contrastive Loss for Anomaly Detection" (AAAI 2023)

Key idea:
- Positive samples should form a compact cluster around the class mean
- Negative samples should be pushed away from the positive cluster
- The "mean-shift" refers to centering the positive embeddings around their mean

Usage:
    python code/fence/models/msc_loss.py  # Runs test cases
"""

import os
import sys
from pathlib import Path

# Suppress warnings
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from tqdm import tqdm


class MSCTripletLoss(nn.Module):
    """
    Mean-Shifted Contrastive Triplet Loss.
    
    For each positive anchor, it is pulled toward the positive class mean,
    while negative samples are pushed away from the mean.
    
    Loss = pos_loss + neg_loss
    where:
    - pos_loss = mean squared distance from positives to mean
    - neg_loss = max(0, margin^2 - distance^2 from negatives to mean)
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        """
        Args:
            margin: Margin for triplet loss
            temperature: Temperature for softmax weighting (not used in this simplified version)
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor,
                pos_mean: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute MSC loss.
        
        Args:
            embeddings: Feature embeddings (N, D)
            labels: Binary labels (N,) where 1 = positive
            pos_mean: Pre-computed positive mean (D,). If None, compute from batch.
            
        Returns:
            Loss scalar
        """
        # Separate positives and negatives
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        pos_embeddings = embeddings[pos_mask]
        neg_embeddings = embeddings[neg_mask]
        
        if len(pos_embeddings) == 0 or len(neg_embeddings) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Compute positive mean
        if pos_mean is None:
            pos_mean = pos_embeddings.mean(dim=0, keepdim=True)
        elif pos_mean.dim() == 1:
            pos_mean = pos_mean.unsqueeze(0)
        
        # Positive loss: pull positives toward the mean (squared distance)
        pos_dist = torch.norm(pos_embeddings - pos_mean, dim=1)
        pos_loss = (pos_dist ** 2).mean()
        
        # Negative loss: push negatives away from the mean (squared distance)
        neg_dist = torch.norm(neg_embeddings - pos_mean, dim=1)
        neg_loss = F.relu(self.margin ** 2 - neg_dist ** 2).mean()
        
        # Total loss
        loss = pos_loss + neg_loss
        
        return loss


class MSCLoss(nn.Module):
    """
    Mean-Shifted Contrastive Loss for FENCE.
    
    This is the full MSC loss that combines:
    1. Pulling positives together (compactness)
    2. Pushing negatives away (separation)
    3. Optional classification loss for stability
    
    Based on: Reiss & Hoshen, "Mean-Shifted Contrastive Loss for Anomaly Detection" (AAAI 2023)
    """
    
    def __init__(self,
                 temperature: float = 0.1,
                 margin: float = 1.0,
                 lambda_class: float = 0.5,
                 use_mean_shift: bool = True,
                 device: str = 'cpu'):
        """
        Args:
            temperature: Temperature for contrastive loss
            margin: Margin for triplet loss
            lambda_class: Weight for classification loss (0 = pure contrastive)
            use_mean_shift: Whether to use mean-shifted version
            device: Device to use
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.lambda_class = lambda_class
        self.use_mean_shift = use_mean_shift
        self.device = device
        
        # Classification loss (BCE)
        self.class_loss = nn.BCEWithLogitsLoss()
        
        # Triplet loss
        self.triplet_loss = MSCTripletLoss(margin=margin, temperature=temperature)
        
        print(f"Initialized MSCLoss:")
        print(f"  Temperature: {temperature}")
        print(f"  Margin: {margin}")
        print(f"  Lambda Class: {lambda_class}")
        print(f"  Mean Shift: {use_mean_shift}")
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor,
                logits: Optional[torch.Tensor] = None,
                pos_mean: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute MSC loss.
        
        Args:
            embeddings: Feature embeddings (N, D)
            labels: Binary labels (N,) where 1 = positive
            logits: Classification logits (N, 1) - optional
            pos_mean: Pre-computed positive mean (D,)
            
        Returns:
            Dictionary with loss components
        """
        # Triplet/contrastive loss
        if self.use_mean_shift:
            loss_triplet = self.triplet_loss(embeddings, labels, pos_mean)
        else:
            # Standard contrastive: pull all positives together, push negatives away
            pos_mask = labels == 1
            neg_mask = labels == 0
            
            pos_embeddings = embeddings[pos_mask]
            neg_embeddings = embeddings[neg_mask]
            
            if len(pos_embeddings) == 0 or len(neg_embeddings) == 0:
                loss_triplet = torch.tensor(0.0, device=embeddings.device)
            else:
                # Compute positive mean (for all positives)
                pos_mean_local = pos_embeddings.mean(dim=0, keepdim=True)
                
                # Positive loss: pull toward mean (squared)
                pos_dist = torch.norm(pos_embeddings - pos_mean_local, dim=1)
                pos_loss = (pos_dist ** 2).mean()
                
                # Negative loss: push away from mean (squared)
                neg_dist = torch.norm(neg_embeddings - pos_mean_local, dim=1)
                neg_loss = F.relu(self.margin ** 2 - neg_dist ** 2).mean()
                
                loss_triplet = pos_loss + neg_loss
        
        # Classification loss
        loss_class = torch.tensor(0.0, device=embeddings.device)
        if logits is not None and self.lambda_class > 0:
            labels_float = labels.float().unsqueeze(1)
            loss_class = self.class_loss(logits, labels_float)
        
        # Combined loss
        loss_total = loss_triplet + self.lambda_class * loss_class
        
        return {
            'total': loss_total,
            'triplet': loss_triplet,
            'class': loss_class
        }
    
    def compute_pos_mean(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean of positive embeddings.
        
        Args:
            embeddings: Feature embeddings (N, D)
            labels: Binary labels (N,)
            
        Returns:
            Positive mean (D,)
        """
        pos_mask = labels == 1
        if not pos_mask.any():
            return torch.zeros(embeddings.shape[1], device=embeddings.device)
        
        pos_embeddings = embeddings[pos_mask]
        return pos_embeddings.mean(dim=0)


class MSCBackboneWithLoss(nn.Module):
    """
    Wrapper that combines a backbone with MSC loss.
    
    This simplifies training by handling the forward pass and loss computation
    in a single module.
    """
    
    def __init__(self,
                 backbone,
                 embedding_dim: int,
                 temperature: float = 0.1,
                 margin: float = 1.0,
                 lambda_class: float = 0.5,
                 use_mean_shift: bool = True,
                 device: str = 'cpu'):
        """
        Args:
            backbone: FENCEBackbone instance
            embedding_dim: Embedding dimension
            temperature: Temperature for contrastive loss
            margin: Margin for triplet loss
            lambda_class: Weight for classification loss
            use_mean_shift: Whether to use mean-shifted version
            device: Device to use
        """
        super().__init__()
        
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.device = device
        
        # MSC loss
        self.msc_loss = MSCLoss(
            temperature=temperature,
            margin=margin,
            lambda_class=lambda_class,
            use_mean_shift=use_mean_shift,
            device=device
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        self.to(device)
        
        print(f"Initialized MSCBackboneWithLoss:")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Temperature: {temperature}")
        print(f"  Margin: {margin}")
        print(f"  Lambda Class: {lambda_class}")
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional loss computation.
        
        Args:
            x: Input images (N, C, H, W)
            labels: Binary labels (N,) - optional
            
        Returns:
            Dictionary with embeddings, logits, and optionally loss
        """
        # Get embeddings from backbone
        embeddings = self.backbone(x)
        
        # Classification logits
        logits = self.classifier(embeddings)
        
        outputs = {
            'embeddings': embeddings,
            'logits': logits
        }
        
        # Compute loss if labels provided
        if labels is not None:
            pos_mean = self.msc_loss.compute_pos_mean(embeddings, labels)
            loss_dict = self.msc_loss(embeddings, labels, logits, pos_mean)
            outputs.update(loss_dict)
        
        return outputs
    
    def compute_pos_mean(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute positive mean from embeddings."""
        return self.msc_loss.compute_pos_mean(embeddings, labels)
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim
    
    def get_lipschitz_constant(self) -> float:
        """Get Lipschitz constant from backbone."""
        return self.backbone.get_lipschitz_constant()


# ============================================================================
# Test Cases
# ============================================================================

def test_msc_triplet_loss():
    """Test MSC triplet loss."""
    print("\n" + "="*60)
    print("TEST: MSC Triplet Loss")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(42)
    dim = 128
    
    # Case 1: Good separation (pos close to mean, neg far)
    pos_mean = torch.ones(dim, device=device)
    pos_embeddings = pos_mean + torch.randn(50, dim, device=device) * 0.1
    neg_embeddings = torch.zeros(50, dim, device=device) + 2.0
    
    embeddings_good = torch.cat([pos_embeddings, neg_embeddings], dim=0)
    labels_good = torch.cat([torch.ones(50, device=device), torch.zeros(50, device=device)], dim=0)
    
    triplet_loss = MSCTripletLoss(margin=1.0)
    loss_good = triplet_loss(embeddings_good, labels_good, pos_mean.unsqueeze(0))
    print(f"  Loss (good separation): {loss_good.item():.4f}")
    
    # Case 2: Poor separation (pos far from mean, neg close)
    pos_far = torch.zeros(dim, device=device)
    pos_embeddings_far = pos_far + torch.randn(50, dim, device=device) * 0.1
    neg_close = torch.ones(dim, device=device)
    neg_embeddings_close = neg_close + torch.randn(50, dim, device=device) * 0.1
    
    embeddings_bad = torch.cat([pos_embeddings_far, neg_embeddings_close], dim=0)
    labels_bad = torch.cat([torch.ones(50, device=device), torch.zeros(50, device=device)], dim=0)
    
    loss_bad = triplet_loss(embeddings_bad, labels_bad, pos_mean.unsqueeze(0))
    print(f"  Loss (poor separation): {loss_bad.item():.4f}")
    
    # Good separation should have lower loss
    assert loss_good < loss_bad, f"Good separation loss ({loss_good:.4f}) should be lower than poor ({loss_bad:.4f})"
    print("  ✓ Loss is lower for well-separated data")
    
    # Check positivity
    assert loss_good > 0, "Loss should be > 0"
    
    print("\n✅ MSC triplet loss tests passed!")


def test_msc_loss():
    """Test full MSC loss."""
    print("\n" + "="*60)
    print("TEST: MSC Loss")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(123)
    n_pos = 30
    n_neg = 30
    dim = 64
    
    # Positives: compact cluster
    pos_embeddings = torch.randn(n_pos, dim, device=device) * 0.15 + 0.5
    
    # Negatives: scattered far
    neg_embeddings = torch.randn(n_neg, dim, device=device) * 0.5
    
    embeddings = torch.cat([pos_embeddings, neg_embeddings], dim=0)
    labels = torch.cat([torch.ones(n_pos, device=device), torch.zeros(n_neg, device=device)], dim=0)
    
    # Logits (dummy)
    logits = torch.randn(len(embeddings), 1, device=device)
    
    # MSC loss
    msc_loss = MSCLoss(temperature=0.1, margin=1.0, lambda_class=0.5, use_mean_shift=True, device=device)
    
    pos_mean = msc_loss.compute_pos_mean(embeddings, labels)
    loss_dict = msc_loss(embeddings, labels, logits, pos_mean)
    
    print(f"  Total loss: {loss_dict['total'].item():.4f}")
    print(f"  Triplet loss: {loss_dict['triplet'].item():.4f}")
    print(f"  Class loss: {loss_dict['class'].item():.4f}")
    
    assert loss_dict['total'].item() > 0, "Loss should be > 0"
    assert loss_dict['triplet'].item() > 0, "Triplet loss should be > 0"
    
    print("\n✅ MSC loss tests passed!")


def test_msc_backbone_wrapper():
    """Test MSCBackboneWithLoss wrapper."""
    print("\n" + "="*60)
    print("TEST: MSC Backbone Wrapper")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dummy backbone (simple MLP for testing)
    class DummyBackbone(nn.Module):
        def __init__(self, dim=64):
            super().__init__()
            self.fc = nn.Linear(3*224*224, dim)
        def forward(self, x):
            x = x.view(x.shape[0], -1)
            return self.fc(x)
        def get_lipschitz_constant(self):
            return 1.0
    
    backbone = DummyBackbone(dim=64)
    
    # MSC wrapper
    wrapper = MSCBackboneWithLoss(
        backbone=backbone,
        embedding_dim=64,
        temperature=0.1,
        margin=1.0,
        lambda_class=0.5,
        use_mean_shift=True,
        device=device
    )
    
    # Dummy input
    batch_size = 16
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    dummy_labels = torch.randint(0, 2, (batch_size,), device=device)
    
    # Forward with loss
    outputs = wrapper(dummy_input, dummy_labels)
    
    print(f"  Embeddings shape: {outputs['embeddings'].shape}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Total loss: {outputs['total'].item():.4f}")
    print(f"  Triplet loss: {outputs['triplet'].item():.4f}")
    print(f"  Class loss: {outputs['class'].item():.4f}")
    
    assert 'embeddings' in outputs, "Missing embeddings"
    assert 'logits' in outputs, "Missing logits"
    assert 'total' in outputs, "Missing total loss"
    
    print("\n✅ MSC backbone wrapper tests passed!")


def test_mean_shift_computation():
    """Test positive mean computation."""
    print("\n" + "="*60)
    print("TEST: Mean Shift Computation")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(456)
    dim = 32
    n_pos = 20
    n_neg = 20
    
    # Positives with different means
    pos_cluster1 = torch.randn(10, dim, device=device) * 0.1 + 0.3
    pos_cluster2 = torch.randn(10, dim, device=device) * 0.1 + 0.7
    pos_embeddings = torch.cat([pos_cluster1, pos_cluster2], dim=0)
    
    neg_embeddings = torch.randn(n_neg, dim, device=device) * 0.3
    
    embeddings = torch.cat([pos_embeddings, neg_embeddings], dim=0)
    labels = torch.cat([torch.ones(n_pos, device=device), torch.zeros(n_neg, device=device)], dim=0)
    
    # Compute mean
    msc_loss = MSCLoss(device=device)
    pos_mean = msc_loss.compute_pos_mean(embeddings, labels)
    
    # Check that mean is between the two clusters
    mean_value = pos_mean[0].item()
    print(f"  Positive mean: {mean_value:.4f}")
    print(f"  Cluster centers: {0.3:.4f}, {0.7:.4f}")
    
    # Mean should be between 0.3 and 0.7
    assert 0.3 < mean_value < 0.7, "Mean should be between clusters"
    print("  ✓ Mean is between clusters")
    
    print("\n✅ Mean shift computation tests passed!")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*60)
    print("RUNNING ALL MSC LOSS TESTS")
    print("="*60)
    
    test_msc_triplet_loss()
    test_msc_loss()
    test_msc_backbone_wrapper()
    test_mean_shift_computation()
    
    print("\n" + "="*60)
    print("🎉 ALL MSC LOSS TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()