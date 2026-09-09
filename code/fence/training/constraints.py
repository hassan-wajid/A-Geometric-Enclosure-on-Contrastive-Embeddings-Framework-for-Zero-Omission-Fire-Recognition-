"""
Feasibility constraints for FENCE training.

Defines the per-sample constraints that enforce zero FNR:
- Each positive sample must be inside its assigned ball with margin m
- Constraint: ||φ(x_i) - c_k||² ≤ (R_k - m)²

Also includes:
- Constraint violation computation
- Lagrange multiplier tracking
- PI controller for dual ascent
- Resilient slack for handling outliers

Usage:
    python code/fence/training/constraints.py  # Runs test cases
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
from typing import Optional, List, Tuple, Dict, Any
from tqdm import tqdm
import numpy as np


class FeasibilityConstraint:
    """
    Per-sample feasibility constraint for FENCE.
    
    Constraint: ||φ(x_i) - c_{a(i)}||² ≤ (R_{a(i)} - m)²
    where a(i) is the ball assigned to positive i.
    
    The violation is: max(0, distance² - (R - m)²)
    """
    
    def __init__(self, margin: float = 0.1, epsilon: float = 0.01):
        """
        Args:
            margin: Margin m (positive samples must be m inside the boundary)
            epsilon: Tolerance for constraint violation
        """
        self.margin = margin
        self.epsilon = epsilon
        self.violations = []
        self.multipliers = []  # Lagrange multipliers λ_i
    
    def compute_violation(self, 
                         embeddings: torch.Tensor,
                         ball_centers: torch.Tensor,
                         ball_radii: torch.Tensor,
                         assignments: torch.Tensor,
                         labels: torch.Tensor) -> torch.Tensor:
        """
        Compute constraint violation for each positive sample.
        
        Args:
            embeddings: Feature embeddings (N, D)
            ball_centers: Ball centers (K, D)
            ball_radii: Ball radii (K,)
            assignments: Ball assignment for each sample (N,)
            labels: Binary labels (N,)
            
        Returns:
            Violation per positive sample (P,)
        """
        pos_mask = labels == 1
        pos_embeddings = embeddings[pos_mask]
        pos_assignments = assignments[pos_mask]
        
        if len(pos_embeddings) == 0:
            return torch.tensor([], device=embeddings.device)
        
        # Get centers and radii for assigned balls
        assigned_centers = ball_centers[pos_assignments]
        assigned_radii = ball_radii[pos_assignments]
        
        # Compute squared distance to assigned center
        diff = pos_embeddings - assigned_centers
        dist_sq = torch.sum(diff ** 2, dim=1)
        
        # Constraint: distance² ≤ (R - m)²
        max_dist_sq = (assigned_radii - self.margin) ** 2
        
        # Violation: max(0, distance² - max_dist_sq)
        violation = F.relu(dist_sq - max_dist_sq)
        
        return violation
    
    def is_feasible(self, 
                   embeddings: torch.Tensor,
                   ball_centers: torch.Tensor,
                   ball_radii: torch.Tensor,
                   assignments: torch.Tensor,
                   labels: torch.Tensor) -> bool:
        """
        Check if all positive samples satisfy the constraint.
        
        Returns:
            True if all violations ≤ epsilon
        """
        violation = self.compute_violation(embeddings, ball_centers, ball_radii, assignments, labels)
        
        if len(violation) == 0:
            return True
        
        return (violation <= self.epsilon).all().item()
    
    def get_max_violation(self, 
                         embeddings: torch.Tensor,
                         ball_centers: torch.Tensor,
                         ball_radii: torch.Tensor,
                         assignments: torch.Tensor,
                         labels: torch.Tensor) -> float:
        """
        Get maximum constraint violation.
        
        Returns:
            Maximum violation value
        """
        violation = self.compute_violation(embeddings, ball_centers, ball_radii, assignments, labels)
        
        if len(violation) == 0:
            return 0.0
        
        return violation.max().item()
    
    def get_mean_violation(self, 
                          embeddings: torch.Tensor,
                          ball_centers: torch.Tensor,
                          ball_radii: torch.Tensor,
                          assignments: torch.Tensor,
                          labels: torch.Tensor) -> float:
        """
        Get mean constraint violation.
        
        Returns:
            Mean violation value
        """
        violation = self.compute_violation(embeddings, ball_centers, ball_radii, assignments, labels)
        
        if len(violation) == 0:
            return 0.0
        
        return violation.mean().item()


class LagrangeMultiplierTracker:
    """
    Tracks and updates Lagrange multipliers for constraints.
    
    Uses PI controller for stable dual ascent.
    """
    
    def __init__(self, 
                 n_samples: int,
                 lr_multiplier: float = 0.01,
                 pi_p: float = 0.1,
                 pi_i: float = 0.01,
                 device: str = 'cpu'):
        """
        Args:
            n_samples: Number of samples (positive only)
            lr_multiplier: Learning rate for multipliers
            pi_p: Proportional gain
            pi_i: Integral gain
            device: Device to use
        """
        self.n_samples = n_samples
        self.lr_multiplier = lr_multiplier
        self.pi_p = pi_p
        self.pi_i = pi_i
        
        # Initialize multipliers
        self.multipliers = torch.zeros(n_samples, device=device)
        self.integral = torch.zeros(n_samples, device=device)
        
        # History for tracking
        self.history = []
    
    def update(self, violation: torch.Tensor, epoch: int = 0) -> torch.Tensor:
        """
        Update multipliers using PI controller.
        
        λ_i ← max(0, λ_i + η * (P * violation + I * integral))
        
        Args:
            violation: Constraint violation per sample
            epoch: Current epoch
            
        Returns:
            Updated multipliers
        """
        if len(violation) == 0:
            return self.multipliers
        
        # Update integral
        self.integral += violation
        
        # PI update
        delta = self.pi_p * violation + self.pi_i * self.integral
        self.multipliers = F.relu(self.multipliers + self.lr_multiplier * delta)
        
        # Store history
        if epoch % 10 == 0:
            self.history.append({
                'epoch': epoch,
                'mean_multiplier': self.multipliers.mean().item(),
                'max_multiplier': self.multipliers.max().item(),
                'mean_violation': violation.mean().item(),
                'max_violation': violation.max().item()
            })
        
        return self.multipliers
    
    def get_multipliers(self) -> torch.Tensor:
        """Get current multipliers."""
        return self.multipliers
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistics about multipliers."""
        if len(self.multipliers) == 0:
            return {'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0}
        
        return {
            'mean': self.multipliers.mean().item(),
            'max': self.multipliers.max().item(),
            'min': self.multipliers.min().item(),
            'std': self.multipliers.std().item()
        }


class ResilientFeasibilityConstraint(FeasibilityConstraint):
    """
    Resilient version of feasibility constraint with slack variables.
    
    Allows small violations with a penalty, and tracks slack as anomaly score.
    """
    
    def __init__(self, margin: float = 0.1, epsilon: float = 0.01, slack_weight: float = 1.0):
        """
        Args:
            margin: Margin m
            epsilon: Tolerance
            slack_weight: Weight for slack in objective
        """
        super().__init__(margin, epsilon)
        self.slack_weight = slack_weight
        self.slack_values = []  # Track slack per sample
    
    def compute_loss_with_slack(self,
                               embeddings: torch.Tensor,
                               ball_centers: torch.Tensor,
                               ball_radii: torch.Tensor,
                               assignments: torch.Tensor,
                               labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute constraint loss with slack variables.
        
        Loss = slack_weight * sum(slack_i²)
        where slack_i = max(0, violation_i - epsilon)
        
        Returns:
            Tuple of (slack_loss, slack_values)
        """
        violation = self.compute_violation(embeddings, ball_centers, ball_radii, assignments, labels)
        
        if len(violation) == 0:
            return torch.tensor(0.0, device=embeddings.device), torch.tensor([])
        
        # Slack = max(0, violation - epsilon)
        slack = F.relu(violation - self.epsilon)
        
        # Slack loss (squared slack)
        slack_loss = self.slack_weight * (slack ** 2).mean()
        
        # Store slack values for tracking
        self.slack_values = slack.detach().cpu().numpy().tolist()
        
        return slack_loss, slack
    
    def get_anomaly_scores(self) -> List[float]:
        """
        Get anomaly scores from slack values.
        High slack = likely mislabeled or anomalous sample.
        """
        return self.slack_values


class FeasibleLearningConstraints:
    """
    Container for all constraints in Feasible Learning.
    
    Manages:
    - Per-sample feasibility constraints
    - Lagrange multipliers
    - Resilient slack
    - Constraint violation tracking
    """
    
    def __init__(self,
                 margin: float = 0.1,
                 epsilon: float = 0.01,
                 lr_multiplier: float = 0.01,
                 pi_p: float = 0.1,
                 pi_i: float = 0.01,
                 slack_weight: float = 1.0,
                 use_slack: bool = True,
                 device: str = 'cpu'):
        """
        Args:
            margin: Margin m
            epsilon: Tolerance for violation
            lr_multiplier: Learning rate for multipliers
            pi_p: Proportional gain
            pi_i: Integral gain
            slack_weight: Weight for slack
            use_slack: Whether to use resilient slack
            device: Device to use
        """
        self.margin = margin
        self.epsilon = epsilon
        self.lr_multiplier = lr_multiplier
        self.pi_p = pi_p
        self.pi_i = pi_i
        self.slack_weight = slack_weight
        self.use_slack = use_slack
        self.device = device
        
        # Constraint and tracker
        self.constraint = ResilientFeasibilityConstraint(margin, epsilon, slack_weight) if use_slack else FeasibilityConstraint(margin, epsilon)
        self.multiplier_tracker = None
        
        # History for tracking
        self.history = []
        
        print(f"Initialized FeasibleLearningConstraints:")
        print(f"  Margin: {margin}")
        print(f"  Epsilon: {epsilon}")
        print(f"  LR Multiplier: {lr_multiplier}")
        print(f"  PI: ({pi_p}, {pi_i})")
        print(f"  Slack: {use_slack}, Weight: {slack_weight}")
    
    def initialize_multipliers(self, n_pos: int):
        """Initialize Lagrange multipliers for positive samples."""
        self.multiplier_tracker = LagrangeMultiplierTracker(
            n_samples=n_pos,
            lr_multiplier=self.lr_multiplier,
            pi_p=self.pi_p,
            pi_i=self.pi_i,
            device=self.device
        )
    
    def compute_violation(self, embeddings: torch.Tensor,
                         ball_centers: torch.Tensor,
                         ball_radii: torch.Tensor,
                         assignments: torch.Tensor,
                         labels: torch.Tensor) -> torch.Tensor:
        """Compute constraint violation."""
        return self.constraint.compute_violation(embeddings, ball_centers, ball_radii, assignments, labels)
    
    def is_feasible(self, embeddings: torch.Tensor,
                   ball_centers: torch.Tensor,
                   ball_radii: torch.Tensor,
                   assignments: torch.Tensor,
                   labels: torch.Tensor) -> bool:
        """Check if all constraints are satisfied."""
        return self.constraint.is_feasible(embeddings, ball_centers, ball_radii, assignments, labels)
    
    def get_max_violation(self, embeddings: torch.Tensor,
                         ball_centers: torch.Tensor,
                         ball_radii: torch.Tensor,
                         assignments: torch.Tensor,
                         labels: torch.Tensor) -> float:
        """Get maximum constraint violation."""
        return self.constraint.get_max_violation(embeddings, ball_centers, ball_radii, assignments, labels)
    
    def get_mean_violation(self, embeddings: torch.Tensor,
                          ball_centers: torch.Tensor,
                          ball_radii: torch.Tensor,
                          assignments: torch.Tensor,
                          labels: torch.Tensor) -> float:
        """Get mean constraint violation."""
        return self.constraint.get_mean_violation(embeddings, ball_centers, ball_radii, assignments, labels)
    
    def compute_constraint_loss(self, embeddings: torch.Tensor,
                               ball_centers: torch.Tensor,
                               ball_radii: torch.Tensor,
                               assignments: torch.Tensor,
                               labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute constraint loss for the primal objective.
        
        Returns:
            Tuple of (constraint_loss, violation)
        """
        violation = self.compute_violation(embeddings, ball_centers, ball_radii, assignments, labels)
        
        if len(violation) == 0:
            return torch.tensor(0.0, device=embeddings.device), violation
        
        if self.use_slack:
            # Use slack loss
            slack_loss, slack = self.constraint.compute_loss_with_slack(
                embeddings, ball_centers, ball_radii, assignments, labels
            )
            return slack_loss, violation
        else:
            # Hard constraint: penalty for violation
            constraint_loss = (violation ** 2).mean()
            return constraint_loss, violation
    
    def update_multipliers(self, violation: torch.Tensor, epoch: int = 0):
        """Update Lagrange multipliers using PI controller."""
        if self.multiplier_tracker is None:
            return
        
        self.multiplier_tracker.update(violation, epoch)
    
    def get_multipliers(self) -> Optional[torch.Tensor]:
        """Get current multipliers."""
        if self.multiplier_tracker is None:
            return None
        return self.multiplier_tracker.get_multipliers()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get constraint statistics."""
        stats = {
            'multipliers': self.multiplier_tracker.get_stats() if self.multiplier_tracker else {},
        }
        return stats


# ============================================================================
# Test Cases
# ============================================================================

def test_feasibility_constraint():
    """Test basic feasibility constraint."""
    print("\n" + "="*60)
    print("TEST: Feasibility Constraint")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dummy data
    torch.manual_seed(42)
    n_samples = 50
    dim = 32
    
    # Ball center and radius
    center = torch.zeros(dim, device=device)
    radius = 1.5  # Increased radius to contain positives
    margin = 0.1
    
    # Positives: keep them inside the ball
    # Sample from a sphere of radius 1.0 (well inside radius 1.5)
    pos_embeddings = torch.randn(n_samples, dim, device=device)
    pos_embeddings = pos_embeddings / torch.norm(pos_embeddings, dim=1, keepdim=True) * 0.8
    
    # Negatives: outside the ball
    neg_embeddings = torch.randn(n_samples, dim, device=device)
    neg_embeddings = neg_embeddings / torch.norm(neg_embeddings, dim=1, keepdim=True) * 2.5
    
    embeddings = torch.cat([pos_embeddings, neg_embeddings], dim=0)
    labels = torch.cat([torch.ones(n_samples, device=device), torch.zeros(n_samples, device=device)], dim=0)
    
    # Assign all positives to ball 0
    assignments = torch.zeros(len(embeddings), dtype=torch.long, device=device)
    
    ball_centers = center.unsqueeze(0)
    ball_radii = torch.tensor([radius], device=device)
    
    # Constraint
    constraint = FeasibilityConstraint(margin=margin, epsilon=0.01)
    
    # Check violation
    violation = constraint.compute_violation(embeddings, ball_centers, ball_radii, assignments, labels)
    
    print(f"  Number of positive samples: {n_samples}")
    print(f"  Mean violation: {violation.mean().item():.6f}")
    print(f"  Max violation: {violation.max().item():.6f}")
    print(f"  Feasible: {constraint.is_feasible(embeddings, ball_centers, ball_radii, assignments, labels)}")
    
    # Positives inside ball should have low violation
    # Since radius=1.5 and positives are at distance ~0.8, violation should be near 0
    # (R - m)² = (1.5 - 0.1)² = 1.96, so distance² ~0.64 should be well under
    assert violation.mean().item() < 0.1, f"Violation should be low for points inside ball, got {violation.mean().item():.4f}"
    print("  ✓ Positives are inside the ball (low violation)")
    
    print("\n✅ Feasibility constraint tests passed!")


def test_lagrange_multiplier_tracker():
    """Test Lagrange multiplier tracker."""
    print("\n" + "="*60)
    print("TEST: Lagrange Multiplier Tracker")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_samples = 10
    
    # Create tracker
    tracker = LagrangeMultiplierTracker(
        n_samples=n_samples,
        lr_multiplier=0.01,
        pi_p=0.1,
        pi_i=0.01,
        device=device
    )
    
    print(f"  Initial multipliers: {tracker.get_multipliers()}")
    
    # Simulate violations
    for epoch in range(20):
        violation = torch.randn(n_samples, device=device) ** 2 * 0.1
        tracker.update(violation, epoch)
    
    stats = tracker.get_stats()
    print(f"  Final multiplier stats: {stats}")
    
    assert stats['mean'] > 0, "Multipliers should increase with violations"
    assert stats['max'] > 0, "Multipliers should increase with violations"
    
    print("\n✅ Lagrange multiplier tracker tests passed!")


def test_resilient_constraint():
    """Test resilient feasibility constraint."""
    print("\n" + "="*60)
    print("TEST: Resilient Constraint")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(123)
    n_pos = 30
    dim = 32
    
    # Ball center and radius
    center = torch.zeros(dim, device=device)
    radius = 1.5
    margin = 0.1
    
    # Positives: some inside, some outside (hard positives)
    pos_inside = torch.randn(n_pos // 2, dim, device=device)
    pos_inside = pos_inside / torch.norm(pos_inside, dim=1, keepdim=True) * 0.7
    
    pos_outside = torch.randn(n_pos // 2, dim, device=device)
    pos_outside = pos_outside / torch.norm(pos_outside, dim=1, keepdim=True) * 2.0
    
    pos_embeddings = torch.cat([pos_inside, pos_outside], dim=0)
    
    # Negatives
    neg_embeddings = torch.randn(n_pos, dim, device=device)
    neg_embeddings = neg_embeddings / torch.norm(neg_embeddings, dim=1, keepdim=True) * 2.5
    
    embeddings = torch.cat([pos_embeddings, neg_embeddings], dim=0)
    labels = torch.cat([torch.ones(n_pos, device=device), torch.zeros(n_pos, device=device)], dim=0)
    
    # Assign all positives to ball 0
    assignments = torch.zeros(len(embeddings), dtype=torch.long, device=device)
    
    ball_centers = center.unsqueeze(0)
    ball_radii = torch.tensor([radius], device=device)
    
    # Resilient constraint
    constraint = ResilientFeasibilityConstraint(margin=margin, epsilon=0.01, slack_weight=1.0)
    
    # Compute slack loss
    slack_loss, slack = constraint.compute_loss_with_slack(
        embeddings, ball_centers, ball_radii, assignments, labels
    )
    
    print(f"  Slack loss: {slack_loss.item():.6f}")
    print(f"  Slack values (first 5): {slack[:5].tolist() if len(slack) > 0 else []}")
    print(f"  Mean slack: {slack.mean().item() if len(slack) > 0 else 0:.6f}")
    
    # Some samples should have slack > 0 (the outside positives)
    assert slack.sum().item() > 0, "Some samples should have slack > 0"
    print("  ✓ Some positives have slack (hard positives detected)")
    
    # Get anomaly scores
    anomaly_scores = constraint.get_anomaly_scores()
    print(f"  Anomaly scores (first 5): {anomaly_scores[:5]}")
    
    assert len(anomaly_scores) > 0, "Anomaly scores should be generated"
    
    print("\n✅ Resilient constraint tests passed!")


def test_full_constraints():
    """Test full feasible learning constraints."""
    print("\n" + "="*60)
    print("TEST: Full Feasible Learning Constraints")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(456)
    n_pos = 40
    dim = 64
    
    # Create ball
    center = torch.zeros(dim, device=device)
    radius = 1.5
    
    # Positives: inside ball
    pos_embeddings = torch.randn(n_pos, dim, device=device)
    pos_embeddings = pos_embeddings / torch.norm(pos_embeddings, dim=1, keepdim=True) * 0.8
    
    # Negatives: outside ball
    neg_embeddings = torch.randn(n_pos, dim, device=device)
    neg_embeddings = neg_embeddings / torch.norm(neg_embeddings, dim=1, keepdim=True) * 2.5
    
    embeddings = torch.cat([pos_embeddings, neg_embeddings], dim=0)
    labels = torch.cat([torch.ones(n_pos, device=device), torch.zeros(n_pos, device=device)], dim=0)
    
    assignments = torch.zeros(len(embeddings), dtype=torch.long, device=device)
    ball_centers = center.unsqueeze(0)
    ball_radii = torch.tensor([radius], device=device)
    
    # Full constraints
    constraints = FeasibleLearningConstraints(
        margin=0.1,
        epsilon=0.01,
        lr_multiplier=0.01,
        pi_p=0.1,
        pi_i=0.01,
        slack_weight=1.0,
        use_slack=True,
        device=device
    )
    
    # Initialize multipliers
    constraints.initialize_multipliers(n_pos)
    
    # Compute constraint loss
    constraint_loss, violation = constraints.compute_constraint_loss(
        embeddings, ball_centers, ball_radii, assignments, labels
    )
    
    print(f"  Constraint loss: {constraint_loss.item():.6f}")
    print(f"  Mean violation: {violation.mean().item():.6f}")
    print(f"  Max violation: {violation.max().item():.6f}")
    print(f"  Feasible: {constraints.is_feasible(embeddings, ball_centers, ball_radii, assignments, labels)}")
    
    # Update multipliers
    constraints.update_multipliers(violation, epoch=0)
    multipliers = constraints.get_multipliers()
    print(f"  Multipliers (first 5): {multipliers[:5].tolist() if multipliers is not None else []}")
    
    assert constraint_loss >= 0, "Constraint loss should be >= 0"
    assert multipliers is not None, "Multipliers should be initialized"
    
    print("\n✅ Full feasible learning constraints tests passed!")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*60)
    print("RUNNING ALL CONSTRAINTS TESTS")
    print("="*60)
    
    test_feasibility_constraint()
    test_lagrange_multiplier_tracker()
    test_resilient_constraint()
    test_full_constraints()
    
    print("\n" + "="*60)
    print("🎉 ALL CONSTRAINTS TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()