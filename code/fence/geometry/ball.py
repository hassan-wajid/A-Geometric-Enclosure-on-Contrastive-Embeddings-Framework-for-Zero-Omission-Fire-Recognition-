"""
Ball geometry for FENCE (Feasible ENclosure with Certified radius).

A single ball in feature space defined by a center and radius.
Supports:
- Distance computation
- Containment checks (with margin)
- Abstention shell detection
- Radius setting from points
- Certified radius via Lipschitz constant

Usage:
    python fence/geometry/ball.py  # Runs test cases
"""

import torch
import numpy as np
from typing import Optional, Tuple, List


class Ball:
    """
    A hypersphere in feature space: B(c, R) = {x : ||x - c|| ≤ R}
    
    Key properties:
    - Center c: center of the ball (tensor)
    - Radius R: radius of the ball (float)
    - Margin m: inward offset for confident predictions and certified radius
    """
    
    def __init__(self, center: torch.Tensor, radius: Optional[float] = None, margin: float = 0.1):
        """
        Initialize a ball.
        
        Args:
            center: Center of the ball (feature vector)
            radius: Initial radius (if None, set to 0)
            margin: Margin for certified radius and abstention shell
        """
        self.center = center.detach().clone() if isinstance(center, torch.Tensor) else torch.tensor(center)
        self.radius = float(radius) if radius is not None else 0.0
        self.margin = float(margin)
        self._assigned_indices = []  # Track which training points are assigned to this ball
    
    def _ensure_2d(self, points: torch.Tensor) -> torch.Tensor:
        """Ensure points is 2D: (N, D). If 1D, add batch dimension."""
        if points.dim() == 1:
            return points.unsqueeze(0)
        return points
    
    def distance(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean distance from points to center.
        
        Args:
            points: Shape (N, D) or (D,) tensor of feature vectors
            
        Returns:
            Distance tensor of shape (N,)
        """
        points = self._ensure_2d(points)
        
        # Ensure points and center are on same device
        if points.device != self.center.device:
            self.center = self.center.to(points.device)
        
        diff = points - self.center
        return torch.norm(diff, dim=1)
    
    def squared_distance(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute squared Euclidean distance from points to center (more efficient).
        
        Args:
            points: Shape (N, D) or (D,) tensor of feature vectors
            
        Returns:
            Squared distance tensor of shape (N,)
        """
        points = self._ensure_2d(points)
        
        if points.device != self.center.device:
            self.center = self.center.to(points.device)
        
        diff = points - self.center
        return torch.sum(diff ** 2, dim=1)
    
    def contains(self, points: torch.Tensor) -> torch.Tensor:
        """
        Check if points are inside the ball (distance ≤ radius).
        
        Args:
            points: Shape (N, D) or (D,) tensor of feature vectors
            
        Returns:
            Boolean tensor of shape (N,)
        """
        points = self._ensure_2d(points)
        return self.distance(points) <= self.radius
    
    def contains_with_margin(self, points: torch.Tensor) -> torch.Tensor:
        """
        Check if points are inside the ball with margin m.
        Points must be at least margin inside the boundary.
        
        distance ≤ radius - margin
        
        Args:
            points: Shape (N, D) or (D,) tensor of feature vectors
            
        Returns:
            Boolean tensor of shape (N,)
        """
        points = self._ensure_2d(points)
        return self.distance(points) <= (self.radius - self.margin)
    
    def in_shell(self, points: torch.Tensor) -> torch.Tensor:
        """
        Check if points are in the abstention shell.
        Shell is the outer band of width margin just inside the boundary.
        
        radius - margin < distance ≤ radius
        
        Args:
            points: Shape (N, D) or (D,) tensor of feature vectors
            
        Returns:
            Boolean tensor of shape (N,)
        """
        points = self._ensure_2d(points)
        dist = self.distance(points)
        return (dist > (self.radius - self.margin)) & (dist <= self.radius)
    
    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        """
        Signed distance from points to the ball boundary.
        Positive = inside, negative = outside.
        
        signed_distance = radius - distance
        
        Args:
            points: Shape (N, D) or (D,) tensor of feature vectors
            
        Returns:
            Signed distance tensor of shape (N,)
        """
        points = self._ensure_2d(points)
        return self.radius - self.distance(points)
    
    def signed_distance_with_margin(self, points: torch.Tensor) -> torch.Tensor:
        """
        Signed distance relative to the margin boundary.
        Positive = inside margin boundary, negative = outside.
        
        signed_distance_margin = (radius - margin) - distance
        
        Args:
            points: Shape (N, D) or (D,) tensor of feature vectors
            
        Returns:
            Signed distance tensor of shape (N,)
        """
        points = self._ensure_2d(points)
        return (self.radius - self.margin) - self.distance(points)
    
    def set_radius_from_points(self, points: torch.Tensor, method: str = 'max', 
                               quantile: float = 0.70) -> float:
        """
        Set radius from a set of points assigned to this ball.
        
        Args:
            points: Shape (N, D) tensor of feature vectors assigned to this ball
            method: 'max' (hard coverage) or 'quantile' (soft coverage)
            quantile: Quantile to use if method='quantile'
            
        Returns:
            The new radius
        """
        points = self._ensure_2d(points)
        
        if len(points) == 0:
            return self.radius
        
        dists = self.distance(points)
        
        if method == 'max':
            self.radius = float(torch.max(dists).item()) + self.margin
        elif method == 'quantile':
            # Add margin to ensure points are inside the boundary
            quantile_val = float(torch.quantile(dists, quantile).item())
            self.radius = quantile_val + self.margin
        else:
            raise ValueError(f"Unknown method: {method}. Use 'max' or 'quantile'.")
        
        return self.radius
    
    def set_center(self, center: torch.Tensor):
        """Update the center of the ball."""
        self.center = center.detach().clone() if isinstance(center, torch.Tensor) else torch.tensor(center)
    
    def get_certified_radius(self, lipschitz_constant: float) -> float:
        """
        Compute the certified radius in input space.
        
        r_certified = margin / L
        
        Args:
            lipschitz_constant: Lipschitz constant L of the backbone
            
        Returns:
            Certified radius in input space
        """
        if lipschitz_constant <= 0:
            return 0.0
        return self.margin / lipschitz_constant
    
    def get_feature_radius(self) -> float:
        """Return the radius in feature space."""
        return self.radius
    
    def assign_points(self, points: torch.Tensor, indices: Optional[List[int]] = None):
        """
        Assign points to this ball (store indices for tracking).
        
        Args:
            points: Points assigned to this ball (for radius computation)
            indices: Original dataset indices of these points
        """
        self._assigned_indices = indices if indices is not None else []
        if len(points) > 0:
            self.set_radius_from_points(points, method='max')
    
    def occupancy(self) -> int:
        """Number of points assigned to this ball."""
        return len(self._assigned_indices)
    
    def to_dict(self) -> dict:
        """Serialize ball to dict."""
        return {
            'center': self.center.cpu().numpy().tolist() if torch.is_tensor(self.center) else self.center,
            'radius': self.radius,
            'margin': self.margin,
            'occupancy': self.occupancy()
        }
    
    def __repr__(self) -> str:
        return f"Ball(center={self.center.shape if torch.is_tensor(self.center) else 'list'}, R={self.radius:.4f}, m={self.margin:.4f}, occ={self.occupancy()})"


# ============================================================================
# Test Cases
# ============================================================================

def test_ball_basic():
    """Test basic ball functionality."""
    print("\n" + "="*60)
    print("TEST: Ball Basic Functionality")
    print("="*60)
    
    # Create a ball centered at origin with radius 1.0, margin 0.1
    center = torch.zeros(3)
    ball = Ball(center, radius=1.0, margin=0.1)
    
    print(f"Ball: {ball}")
    
    # Test points
    inside = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.9]])
    outside = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.1, 0.0, 0.0]])
    on_boundary = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    
    # Contains
    inside_mask = ball.contains(inside)
    assert inside_mask.all(), "All inside points should be contained"
    print(f"  ✓ All inside points contained")
    
    outside_mask = ball.contains(outside)
    assert not outside_mask.any(), "No outside points should be contained"
    print(f"  ✓ No outside points contained")
    
    # Contains with margin
    margin_inside = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    margin_outside = torch.tensor([[0.95, 0.0, 0.0], [0.0, 0.95, 0.0]])
    
    margin_mask = ball.contains_with_margin(margin_inside)
    assert margin_mask.all(), "Points within margin should be contained"
    print(f"  ✓ Points within margin contained")
    
    margin_mask_out = ball.contains_with_margin(margin_outside)
    assert not margin_mask_out.any(), "Points in shell should NOT be contained with margin"
    print(f"  ✓ Points in shell correctly rejected")
    
    # Shell detection
    shell_inside = torch.tensor([[0.95, 0.0, 0.0], [0.0, 0.95, 0.0], [0.0, 0.0, 0.95]])
    shell_mask = ball.in_shell(shell_inside)
    assert shell_mask.all(), "Points in shell should be detected"
    print(f"  ✓ Shell detection working")
    
    # Signed distance - test with 2D tensor
    signed = ball.signed_distance(inside)
    assert (signed > 0).all(), "Inside points should have positive signed distance"
    print(f"  ✓ Signed distance positive for inside points")
    
    signed = ball.signed_distance(outside)
    assert (signed < 0).all(), "Outside points should have negative signed distance"
    print(f"  ✓ Signed distance negative for outside points")
    
    signed = ball.signed_distance(on_boundary)
    assert torch.allclose(signed, torch.zeros_like(signed), atol=1e-6), "Boundary points should have zero signed distance"
    print(f"  ✓ Signed distance zero for boundary points")
    
    # Test 1D input (single point as vector)
    single_point = torch.tensor([0.5, 0.0, 0.0])
    dist = ball.distance(single_point)
    assert dist.shape == torch.Size([1]), "1D input should return 1D output"
    assert dist.item() == 0.5, "Distance should be 0.5"
    print(f"  ✓ 1D input handling works")
    
    print("\n✅ All basic tests passed!")


def test_radius_setting():
    """Test radius setting from points."""
    print("\n" + "="*60)
    print("TEST: Radius Setting")
    print("="*60)
    
    center = torch.zeros(3)
    ball = Ball(center, radius=0.0, margin=0.1)
    
    # Points at various distances
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.8, 0.0],
        [0.0, 0.0, 1.0],
    ])
    
    # Max method
    ball.set_radius_from_points(points, method='max')
    expected_max = 1.0 + 0.1  # max distance + margin
    assert abs(ball.radius - expected_max) < 1e-6, f"Expected {expected_max}, got {ball.radius}"
    print(f"  ✓ Max radius: {ball.radius:.4f} (expected {expected_max:.4f})")
    
    # Check all points are contained
    assert ball.contains(points).all(), "All points should be contained after setting radius"
    print(f"  ✓ All points contained")
    
    # Quantile method
    ball2 = Ball(center, radius=0.0, margin=0.1)
    ball2.set_radius_from_points(points, method='quantile', quantile=0.95)
    # Should be slightly less than max
    assert ball2.radius < ball.radius, "Quantile radius should be <= max radius"
    print(f"  ✓ Quantile radius: {ball2.radius:.4f} (less than max {ball.radius:.4f})")
    
    print("\n✅ All radius setting tests passed!")


def test_certified_radius():
    """Test certified radius computation."""
    print("\n" + "="*60)
    print("TEST: Certified Radius")
    print("="*60)
    
    center = torch.zeros(3)
    margin = 0.1
    ball = Ball(center, radius=1.0, margin=margin)
    
    # Various Lipschitz constants
    L_values = [1.0, 2.0, 5.0, 10.0]
    expected_r = [0.1, 0.05, 0.02, 0.01]
    
    for L, expected in zip(L_values, expected_r):
        r = ball.get_certified_radius(L)
        assert abs(r - expected) < 1e-6, f"For L={L}, expected {expected}, got {r}"
        print(f"  ✓ L={L:.0f} → r={r:.4f}")
    
    print("\n✅ All certified radius tests passed!")


def test_margin_consistency():
    """Test that margin is consistent across methods."""
    print("\n" + "="*60)
    print("TEST: Margin Consistency")
    print("="*60)
    
    margin = 0.15
    center = torch.zeros(3)
    ball = Ball(center, radius=1.0, margin=margin)
    
    # A point exactly at radius - margin (should be inside with margin)
    point_inside = torch.tensor([[0.0, 0.0, 0.85]])  # distance = 0.85 = R - m
    point_shell = torch.tensor([[0.0, 0.0, 0.92]])   # distance = 0.92 = R - 0.08 (in shell)
    point_outside = torch.tensor([[0.0, 0.0, 1.05]]) # distance = 1.05 > R
    
    assert ball.contains_with_margin(point_inside).item(), "Point at R-m should be inside with margin"
    assert not ball.contains_with_margin(point_shell).item(), "Point in shell should NOT be inside with margin"
    assert ball.in_shell(point_shell).item(), "Point in shell should be detected"
    assert not ball.in_shell(point_inside).item(), "Point inside margin should NOT be in shell"
    assert not ball.contains(point_outside).item(), "Outside point should not be contained"
    
    print(f"  ✓ Margin={margin:.2f} works consistently")
    print(f"  ✓ Shell width = {margin:.2f} (absolute distance)")
    
    print("\n✅ All margin consistency tests passed!")


def test_device_compatibility():
    """Test that ball works on different devices."""
    print("\n" + "="*60)
    print("TEST: Device Compatibility")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("  ⚠ CUDA not available, skipping device test")
        return
    
    center = torch.zeros(3).cuda()
    ball = Ball(center, radius=1.0, margin=0.1)
    
    # Points on CPU
    points_cpu = torch.tensor([[0.5, 0.0, 0.0]])
    
    # Distance should auto-move center to CPU
    dist = ball.distance(points_cpu)
    assert dist.device == points_cpu.device, "Distance should be on CPU"
    print(f"  ✓ CPU→CPU works")
    
    # Points on GPU
    points_gpu = torch.tensor([[0.5, 0.0, 0.0]]).cuda()
    dist = ball.distance(points_gpu)
    assert dist.device == points_gpu.device, "Distance should be on GPU"
    print(f"  ✓ GPU→GPU works")
    
    print("\n✅ All device compatibility tests passed!")


def test_serialization():
    """Test serialization to dict."""
    print("\n" + "="*60)
    print("TEST: Serialization")
    print("="*60)
    
    center = torch.tensor([1.0, 2.0, 3.0])
    ball = Ball(center, radius=2.5, margin=0.2)
    ball._assigned_indices = [0, 1, 2, 3, 4]
    
    d = ball.to_dict()
    print(f"  Serialized: {d}")
    
    # Check fields
    assert 'center' in d
    assert 'radius' in d
    assert 'margin' in d
    assert 'occupancy' in d
    assert d['radius'] == 2.5
    assert d['margin'] == 0.2
    assert d['occupancy'] == 5
    
    print("  ✓ All fields present and correct")
    
    print("\n✅ All serialization tests passed!")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*60)
    print("RUNNING ALL BALL TESTS")
    print("="*60)
    
    test_ball_basic()
    test_radius_setting()
    test_certified_radius()
    test_margin_consistency()
    test_device_compatibility()
    test_serialization()
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()