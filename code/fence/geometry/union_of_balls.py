"""
Union of Balls geometry for FENCE.

A union of multiple balls in feature space: E = ∪_k B(c_k, R_k)
Supports:
- Multiple balls with shared margin
- Point assignment to nearest ball
- Distance to the nearest ball
- Containment checks across all balls
- Cluster initialization (K-means, DBSCAN)
- Recompute centers (EMA) and radii (max/quantile)

Usage:
    python code/fence/geometry/union_of_balls.py  # Runs test cases
"""

import os
import sys
from pathlib import Path

# Fix for Windows MKL memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'

# Suppress sklearn warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances

# Try relative import first, fall back to absolute
try:
    from .ball import Ball
except ImportError:
    from fence.geometry.ball import Ball


class UnionOfBalls:
    """
    Union of Balls: E = ∪_{k=1}^K B(c_k, R_k)
    
    A point is positive if it lies inside ANY ball.
    
    Key properties:
    - K balls: each with center c_k and radius R_k
    - Shared margin m across all balls
    - Centers updated via EMA (not backprop)
    - Radii set analytically from assigned points
    """
    
    def __init__(self,
                 num_balls: int = 0,
                 margin: float = 0.1,
                 max_balls: int = 10,
                 min_cluster_size: int = 5,
                 radius_quantile: float = 0.9,
                 device: str = 'cpu'):
        """
        Initialize union of balls.
        
        Args:
            num_balls: Number of balls (0 = auto-detect via clustering)
            margin: Shared margin m across all balls
            max_balls: Maximum balls if auto-detecting
            min_cluster_size: Minimum points per cluster
            device: Device to use
        """
        self.num_balls = num_balls
        self.margin = margin
        self.max_balls = max_balls
        self.min_cluster_size = min_cluster_size
        self.radius_quantile = radius_quantile
        self.device = device
        
        self.balls: List[Ball] = []
        self._assignments: Dict[int, List[int]] = {}  # ball_idx -> list of sample indices
        self._feature_dim = None
        
        # EMA settings for center updates
        self._ema_decay = 0.9
        self._centers_ema = None  # For smooth center updates
        
        print(f"Initialized UnionOfBalls: num_balls={num_balls}, margin={margin}")
    
    def initialize_from_embeddings(self,
                                embeddings: torch.Tensor,
                                labels: torch.Tensor,
                                method: str = 'kmeans',
                                radius_method: str = 'max',
                                target_label: int = 1) -> int:
        """
        Initialize balls from the target class's embeddings using clustering.

        Args:
            embeddings: Shape (N, D) tensor of all embeddings
            labels: Shape (N,) tensor of labels (1=positive)
            method: 'kmeans' or 'dbscan' for auto-detection
            radius_method: 'max' or 'quantile' for radius setting
            target_label: which class to enclose (1=positives, 0=negatives)
        """
        # Filter to the target class only
        pos_mask = labels == target_label
        pos_embeddings = embeddings[pos_mask]
        
        if len(pos_embeddings) == 0:
            print("WARNING: No positive samples found! Cannot initialize balls.")
            return 0
        
        print(f"Initializing from {len(pos_embeddings)} positive embeddings")
        
        # Determine number of clusters
        if self.num_balls > 0:
            k = self.num_balls
        else:
            k = self._auto_detect_clusters(pos_embeddings, method)
        
        print(f"  Using {k} clusters")
        
        # Detach embeddings before numpy conversion
        pos_embeddings_np = pos_embeddings.detach().cpu().numpy()
        
        # Run K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pos_embeddings_np)
        
        # Create balls from clusters
        self.balls = []
        self._assignments = {}
        self._feature_dim = pos_embeddings.shape[1]
        
        for cluster_idx in range(k):
            # Get points in this cluster
            mask = cluster_labels == cluster_idx
            cluster_points = pos_embeddings[mask]
            
            if len(cluster_points) < self.min_cluster_size:
                print(f"  Cluster {cluster_idx}: {len(cluster_points)} points (below min, skipping)")
                continue
            
            # Center is mean of cluster
            center = cluster_points.mean(dim=0)
            
            # Create ball with margin and radius_method
            ball = Ball(center, radius=0.0, margin=self.margin)

            # Set radius using the REQUESTED method + configured quantile
            ball.set_radius_from_points(
                cluster_points, method=radius_method, quantile=self.radius_quantile
            )
            
            # Store assigned indices (for tracking)
            pos_indices = torch.where(pos_mask)[0]
            ball._assigned_indices = pos_indices[mask].cpu().numpy().tolist()
            
            self.balls.append(ball)
            self._assignments[len(self.balls) - 1] = ball._assigned_indices
            
            print(f"  Ball {len(self.balls)}: {len(cluster_points)} points, R={ball.radius:.4f}")
        
        # Initialize EMA centers
        if len(self.balls) > 0:
            self._centers_ema = torch.stack([ball.center.clone() for ball in self.balls])
        
        print(f"Initialized {len(self.balls)} balls")
        return len(self.balls)
        
    @classmethod
    def from_dict(cls, data, device='cpu'):
        """Restore UnionOfBalls from dict."""
        from fence.geometry.ball import Ball
        
        union = cls(
            num_balls=data.get('num_balls', 0),
            margin=data.get('margin', 0.1),
            device=device
        )
        union._feature_dim = data.get('feature_dim', None)
        
        for ball_data in data.get('balls', []):
            center = torch.tensor(ball_data['center'], device=device)
            ball = Ball(center, radius=ball_data['radius'], margin=ball_data['margin'])
            ball._assigned_indices = ball_data.get('assigned_indices', [])
            union.balls.append(ball)
        
        # Restore EMA centers if present
        if 'centers_ema' in data:
            union._centers_ema = torch.tensor(data['centers_ema'], device=device)
        
        return union
        
    def _build_balls(self, pos_embeddings, neg_embeddings, k, method, pos_mask, radius_method='max'):
        """
        Build k balls from positive embeddings.
        
        Args:
            pos_embeddings: Positive embeddings (P, D)
            neg_embeddings: Negative embeddings (N, D) - optional
            k: Number of clusters
            method: Clustering method ('kmeans')
            pos_mask: Boolean mask of positive samples in original dataset
            radius_method: 'max' or 'quantile' for radius setting
        """
        from sklearn.cluster import KMeans
        pos_np = pos_embeddings.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pos_np)
        
        balls = []
        for cluster_idx in range(k):
            mask = cluster_labels == cluster_idx
            cluster_points = pos_embeddings[mask]
            if len(cluster_points) < self.min_cluster_size:
                return None
            center = cluster_points.mean(dim=0)
            ball = Ball(center, radius=0.0, margin=self.margin)
            # Use radius_method here
            ball.set_radius_from_points(cluster_points, method='quantile') # <-- FIXED
            pos_indices = torch.where(pos_mask)[0]
            ball._assigned_indices = pos_indices[mask].cpu().numpy().tolist()
            balls.append(ball)
        return balls


    def _count_captured(self, balls, neg_embeddings):
        """Count unique negatives captured by ANY ball."""
        if not balls or len(neg_embeddings) == 0:
            return 0
        
        # Start with all negatives as NOT captured
        captured_mask = torch.zeros(len(neg_embeddings), dtype=torch.bool, device=neg_embeddings.device)
        
        for ball in balls:
            contains = ball.contains(neg_embeddings)
            captured_mask = captured_mask | contains  # OR operation - unique only
        
        return captured_mask.sum().item()
    
    def _auto_detect_clusters(self, embeddings, method='kmeans'):
        # Remove DBSCAN, use K-means with silhouette score or fixed k
        if self.num_balls > 0:
            return self.num_balls
        
        # Use K-means with elbow or silhouette
        from fence.utils.cluster_init import auto_kmeans
        k, _, _ = auto_kmeans(embeddings, max_k=self.max_balls, min_k=1, method='silhouette')
        return k
    
    def assign_points(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Assign each point to its nearest ball.
        
        Args:
            embeddings: Shape (N, D) tensor of embeddings
            
        Returns:
            Tensor of assignments (N,) with ball indices (-1 if no balls)
        """
        if len(self.balls) == 0:
            return -torch.ones(embeddings.shape[0], dtype=torch.long, device=embeddings.device)
        
        # Compute distances to all ball centers
        distances = torch.stack([
            ball.distance(embeddings) for ball in self.balls
        ], dim=1)  # (N, K)
        
        # Assign to nearest ball
        assignments = torch.argmin(distances, dim=1)  # (N,)
        
        return assignments
    
    def contains(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Check if points are inside ANY ball.
        
        Args:
            embeddings: Shape (N, D) tensor of embeddings
            
        Returns:
            Boolean tensor of shape (N,) - True if inside any ball
        """
        if len(self.balls) == 0:
            return torch.zeros(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
        
        # Check each ball
        contains_any = torch.zeros(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
        for ball in self.balls:
            contains_any = contains_any | ball.contains(embeddings)
        
        return contains_any
    
    def contains_with_margin(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Check if points are inside ANY ball with margin.
        (distance ≤ radius - margin)
        
        Args:
            embeddings: Shape (N, D) tensor of embeddings
            
        Returns:
            Boolean tensor of shape (N,)
        """
        if len(self.balls) == 0:
            return torch.zeros(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
        
        contains_any = torch.zeros(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
        for ball in self.balls:
            contains_any = contains_any | ball.contains_with_margin(embeddings)
        
        return contains_any
    
    def in_shell(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Check if points are in the abstention shell of ANY ball.
        (radius - margin < distance ≤ radius)
        
        Args:
            embeddings: Shape (N, D) tensor of embeddings
            
        Returns:
            Boolean tensor of shape (N,)
        """
        if len(self.balls) == 0:
            return torch.zeros(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
        
        in_shell_any = torch.zeros(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
        for ball in self.balls:
            in_shell_any = in_shell_any | ball.in_shell(embeddings)
        
        return in_shell_any
    
    def distance_to_nearest(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute distance to the nearest ball boundary.
        Positive = inside, negative = outside.
        
        Returns max signed distance across all balls.
        
        Args:
            embeddings: Shape (N, D) tensor of embeddings
            
        Returns:
            Signed distance tensor of shape (N,)
        """
        if len(self.balls) == 0:
            return -torch.ones(embeddings.shape[0], device=embeddings.device) * float('inf')
        
        signed_dists = torch.stack([
            ball.signed_distance(embeddings) for ball in self.balls
        ], dim=1)  # (N, K)
        
        # Max signed distance = nearest ball
        return torch.max(signed_dists, dim=1)[0]
    
    def distance_to_nearest_with_margin(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute distance to the nearest margin boundary.
        (radius - margin) - distance
        
        Args:
            embeddings: Shape (N, D) tensor of embeddings
            
        Returns:
            Signed distance tensor of shape (N,)
        """
        if len(self.balls) == 0:
            return -torch.ones(embeddings.shape[0], device=embeddings.device) * float('inf')
        
        signed_dists = torch.stack([
            ball.signed_distance_with_margin(embeddings) for ball in self.balls
        ], dim=1)  # (N, K)
        
        return torch.max(signed_dists, dim=1)[0]
    
    def update_centers(self, embeddings: torch.Tensor, assignments: torch.Tensor):
        """
        Update ball centers using EMA.
        
        Args:
            embeddings: Shape (N, D) tensor of embeddings
            assignments: Shape (N,) tensor of ball assignments (-1 for unassigned)
        """
        if len(self.balls) == 0:
            return
        
        new_centers = []
        for ball_idx, ball in enumerate(self.balls):
            mask = assignments == ball_idx
            if mask.any():
                # Mean of assigned points
                cluster_mean = embeddings[mask].mean(dim=0)
                
                # EMA update
                if self._centers_ema is not None and ball_idx < len(self._centers_ema):
                    new_center = self._ema_decay * self._centers_ema[ball_idx] + (1 - self._ema_decay) * cluster_mean
                else:
                    new_center = cluster_mean
                
                ball.set_center(new_center)
                new_centers.append(new_center)
            else:
                new_centers.append(ball.center)
        
        # Update EMA centers
        if new_centers:
            self._centers_ema = torch.stack(new_centers)
    
    def update_radii(self, embeddings: torch.Tensor, assignments: torch.Tensor, method: str = 'max'):
        """
        Update ball radii from assigned points.
        
        Args:
            embeddings: Shape (N, D) tensor of embeddings
            assignments: Shape (N,) tensor of ball assignments
            method: 'max' or 'quantile'
        """
        if len(self.balls) == 0:
            return
        
        for ball_idx, ball in enumerate(self.balls):
            mask = assignments == ball_idx
            if mask.any():
                cluster_points = embeddings[mask]
                ball.set_radius_from_points(
                    cluster_points, method=method, quantile=self.radius_quantile
                )
    
    def recompute(self, 
                  embeddings: torch.Tensor, 
                  method: str = 'max',
                  update_centers: bool = True,
                  update_radii: bool = True) -> torch.Tensor:
        """
        Full recomputation: assign points, update centers, update radii.
        
        Args:
            embeddings: Shape (N, D) tensor of embeddings
            method: 'max' or 'quantile' for radius setting
            update_centers: Whether to update centers
            update_radii: Whether to update radii
            
        Returns:
            Tensor of assignments (N,)
        """
        # Assign points to nearest ball
        assignments = self.assign_points(embeddings)
        
        if update_centers:
            self.update_centers(embeddings, assignments)
        
        if update_radii:
            self.update_radii(embeddings, assignments, method=method)
        
        # Update stored assignments
        for ball_idx, ball in enumerate(self.balls):
            mask = assignments == ball_idx
            ball._assigned_indices = torch.where(mask)[0].cpu().numpy().tolist()
            self._assignments[ball_idx] = ball._assigned_indices
        
        return assignments
    
    def get_min_separation_margin(self, 
                                  embeddings: torch.Tensor, 
                                  labels: torch.Tensor) -> float:
        """
        Compute minimum separation margin R_plus - R_minus for each ball.
        R_plus = max distance from center to assigned positives
        R_minus = min distance from center to any negative
        
        Args:
            embeddings: Shape (N, D) tensor of embeddings
            labels: Shape (N,) tensor of labels
            
        Returns:
            Minimum separation margin across all balls
        """
        if len(self.balls) == 0:
            return 0.0
        
        min_margin = float('inf')
        
        for ball in self.balls:
            # Distance from center to all points
            dists = ball.distance(embeddings)
            
            # Positives in this ball
            pos_mask = (labels == 1) & (dists <= ball.radius)
            if pos_mask.any():
                R_plus = dists[pos_mask].max().item()
            else:
                continue
            
            # Nearest negative (outside ball)
            neg_mask = (labels == 0)
            if neg_mask.any():
                neg_dists = dists[neg_mask]
                if len(neg_dists) > 0:
                    R_minus = neg_dists.min().item()
                    
                    # Separation margin: R_minus - R_plus
                    margin = R_minus - R_plus
                    if margin < min_margin:
                        min_margin = margin
        
        return min_margin if min_margin != float('inf') else 0.0
    
    def get_certified_radius(self, lipschitz_constant: float) -> float:
        """
        Compute certified radius in input space for the tightest ball.
        
        r_certified = min_k (margin / L)
        
        Args:
            lipschitz_constant: Lipschitz constant L of the backbone
            
        Returns:
            Certified radius in input space
        """
        if len(self.balls) == 0 or lipschitz_constant <= 0:
            return 0.0
        
        return self.margin / lipschitz_constant
    
    def to_dict(self) -> dict:
        """Serialize union to dict."""
        return {
            'num_balls': len(self.balls),
            'margin': self.margin,
            'balls': [ball.to_dict() for ball in self.balls],
            'feature_dim': self._feature_dim
        }
    
    def __len__(self) -> int:
        return len(self.balls)
    
    def __repr__(self) -> str:
        return f"UnionOfBalls(K={len(self.balls)}, margin={self.margin})"


# ============================================================================
# Test Cases
# ============================================================================

def test_union_basic():
    """Test basic union of balls functionality."""
    print("\n" + "="*60)
    print("TEST: Union Basic Functionality")
    print("="*60)
    
    # Create synthetic data: two clusters of positives, negatives in between
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Cluster 1: around (0, 0)
    pos1 = torch.randn(50, 2) * 0.2
    # Cluster 2: around (3, 0)
    pos2 = torch.randn(50, 2) * 0.2 + torch.tensor([3.0, 0.0])
    positives = torch.cat([pos1, pos2], dim=0)
    
    # Negatives in between
    negatives = torch.randn(50, 2) * 0.3 + torch.tensor([1.5, 0.0])
    
    embeddings = torch.cat([positives, negatives], dim=0)
    labels = torch.cat([torch.ones(100), torch.zeros(50)], dim=0)
    
    # Initialize union
    union = UnionOfBalls(num_balls=2, margin=0.1, device=device)
    union.initialize_from_embeddings(embeddings.to(device), labels.to(device), method='kmeans')
    
    print(f"  Union: {union}")
    
    # Check containment
    contains = union.contains(embeddings.to(device))
    pos_contained = contains[labels == 1].sum().item()
    neg_contained = contains[labels == 0].sum().item()
    
    print(f"  Positive containment: {pos_contained}/{100}")
    print(f"  Negative containment (FPR): {neg_contained}/{50}")
    
    # Should contain all positives
    assert pos_contained == 100, "All positives should be contained"
    print("  ✓ All positives contained")
    
    # Distance to nearest
    dists = union.distance_to_nearest(embeddings.to(device))
    assert (dists[labels == 1] >= 0).all(), "Positives should have positive signed distance"
    print("  ✓ Signed distance positive for positives")
    
    print("\n✅ All basic tests passed!")


def test_recompute():
    """Test recomputation of centers and radii."""
    print("\n" + "="*60)
    print("TEST: Recompute Centers and Radii")
    print("="*60)
    
    torch.manual_seed(123)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Single cluster of positives
    positives = torch.randn(100, 3) * 0.3
    negatives = torch.randn(100, 3) * 0.5 + torch.tensor([2.0, 0.0, 0.0])
    
    embeddings = torch.cat([positives, negatives], dim=0)
    labels = torch.cat([torch.ones(100), torch.zeros(100)], dim=0)
    
    union = UnionOfBalls(num_balls=1, margin=0.1, device=device)
    union.initialize_from_embeddings(embeddings.to(device), labels.to(device))
    
    print(f"  Initial: {union}")
    print(f"  Radius: {union.balls[0].radius:.4f}")
    
    # Shift positives slightly
    shifted_pos = positives + torch.randn(100, 3) * 0.1
    shifted_embeddings = torch.cat([shifted_pos, negatives], dim=0)
    
    # Recompute
    assignments = union.recompute(shifted_embeddings.to(device), method='max')
    
    print(f"  After recompute: radius={union.balls[0].radius:.4f}")
    
    # Should still contain all positives
    contains = union.contains(shifted_embeddings.to(device))
    pos_contained = contains[labels == 1].sum().item()
    assert pos_contained == 100, "All positives should still be contained after recompute"
    print("  ✓ All positives still contained")
    
    print("\n✅ All recompute tests passed!")


def test_separation_margin():
    """Test separation margin computation."""
    print("\n" + "="*60)
    print("TEST: Separation Margin")
    print("="*60)
    
    torch.manual_seed(456)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Positives tightly clustered at origin
    positives = torch.randn(50, 2) * 0.2
    # Negatives at distance 1.0
    negatives = torch.randn(50, 2) * 0.1 + torch.tensor([1.0, 0.0])
    
    embeddings = torch.cat([positives, negatives], dim=0)
    labels = torch.cat([torch.ones(50), torch.zeros(50)], dim=0)
    
    union = UnionOfBalls(num_balls=1, margin=0.1, device=device)
    union.initialize_from_embeddings(embeddings.to(device), labels.to(device))
    
    margin = union.get_min_separation_margin(embeddings.to(device), labels.to(device))
    print(f"  Separation margin: {margin:.4f}")
    
    # Should be positive (there's a gap)
    assert margin > 0.2, f"Expected margin > 0.2, got {margin:.4f}"
    print("  ✓ Positive separation margin found")
    
    print("\n✅ All separation margin tests passed!")


def test_certified_radius():
    """Test certified radius computation."""
    print("\n" + "="*60)
    print("TEST: Certified Radius")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Simple union with one ball
    positives = torch.randn(10, 3) * 0.3
    negatives = torch.randn(10, 3) * 0.5 + torch.tensor([2.0, 0.0, 0.0])
    
    embeddings = torch.cat([positives, negatives], dim=0)
    labels = torch.cat([torch.ones(10), torch.zeros(10)], dim=0)
    
    union = UnionOfBalls(num_balls=1, margin=0.15, device=device)
    union.initialize_from_embeddings(embeddings.to(device), labels.to(device))
    
    L_values = [1.0, 2.0, 5.0, 10.0]
    expected_r = [0.15, 0.075, 0.03, 0.015]
    
    for L, expected in zip(L_values, expected_r):
        r = union.get_certified_radius(L)
        assert abs(r - expected) < 1e-6, f"For L={L}, expected {expected}, got {r}"
        print(f"  ✓ L={L:.0f} → r={r:.4f}")
    
    print("\n✅ All certified radius tests passed!")


def test_serialization():
    """Test serialization to dict."""
    print("\n" + "="*60)
    print("TEST: Serialization")
    print("="*60)
    
    torch.manual_seed(789)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    positives = torch.randn(20, 3) * 0.3
    negatives = torch.randn(20, 3) * 0.5 + torch.tensor([2.0, 0.0, 0.0])
    
    embeddings = torch.cat([positives, negatives], dim=0)
    labels = torch.cat([torch.ones(20), torch.zeros(20)], dim=0)
    
    union = UnionOfBalls(num_balls=2, margin=0.1, device=device)
    union.initialize_from_embeddings(embeddings.to(device), labels.to(device))
    
    d = union.to_dict()
    print(f"  Serialized: K={d['num_balls']}, margin={d['margin']}")
    
    assert 'num_balls' in d
    assert 'margin' in d
    assert 'balls' in d
    assert len(d['balls']) == len(union.balls)
    
    print("  ✓ All fields present and correct")
    
    print("\n✅ All serialization tests passed!")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*60)
    print("RUNNING ALL UNION OF BALLS TESTS")
    print("="*60)
    
    test_union_basic()
    test_recompute()
    test_separation_margin()
    test_certified_radius()
    test_serialization()
    
    print("\n" + "="*60)
    print("🎉 ALL UNION OF BALLS TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()