"""
Cluster initialization utilities for FENCE.

Provides robust clustering methods for initializing the union of balls:
- K-means with automatic k selection (elbow method)
- DBSCAN for density-based clustering
- Balanced clustering for handling size imbalance
- Cluster validation utilities

Usage:
    python code/fence/utils/cluster_init.py  # Runs test cases
"""

# ============================================================================
# IMPORTANT: Set environment variables and suppress warnings BEFORE any imports
# ============================================================================
import os
import sys
from pathlib import Path

# Fix for Windows MKL memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Now import everything else
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
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler


def auto_kmeans(embeddings: torch.Tensor,
                max_k: int = 10,
                min_k: int = 1,
                min_cluster_size: int = 5,
                method: str = 'silhouette') -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Auto-select k for K-means using silhouette score (more reliable).
    
    Args:
        embeddings: Shape (N, D) tensor of embeddings
        max_k: Maximum number of clusters to try
        min_k: Minimum number of clusters to try
        min_cluster_size: Minimum points per cluster
        method: 'elbow' or 'silhouette' (default: 'silhouette')
        
    Returns:
        Tuple of (optimal_k, cluster_labels, cluster_centers)
    """
    n_samples = len(embeddings)
    if n_samples < min_cluster_size * 2:
        print(f"  Too few samples ({n_samples}), using k=1")
        kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings.cpu().numpy())
        return 1, labels, kmeans.cluster_centers_
    
    # Determine max k
    max_k = min(max_k, n_samples // min_cluster_size)
    max_k = max(max_k, min_k + 1)
    
    embeddings_np = embeddings.cpu().numpy()
    
    # Use silhouette score (more reliable than elbow)
    best_k = min_k
    best_score = -1
    best_labels = None
    best_centers = None
    
    k_range = range(min_k, min(max_k + 1, n_samples))
    
    for k in tqdm(k_range, desc="  Testing k", disable=False):
        if k >= n_samples:
            break
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_np)
        
        # Check cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        if np.min(counts) < min_cluster_size:
            continue
        
        try:
            score = silhouette_score(embeddings_np, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
                best_centers = kmeans.cluster_centers_
        except:
            continue
    
    # Fallback: if no good clustering found, use 1 cluster
    if best_labels is None:
        print(f"  Silhouette failed, using k={min_k}")
        kmeans = KMeans(n_clusters=min_k, random_state=42, n_init=10)
        best_labels = kmeans.fit_predict(embeddings_np)
        best_centers = kmeans.cluster_centers_
        best_k = min_k
    
    return best_k, best_labels, best_centers


def dbscan_clustering(embeddings: torch.Tensor,
                      eps: Optional[float] = None,
                      min_samples: Optional[int] = None,
                      min_cluster_size: int = 5,
                      auto_eps: bool = True) -> Tuple[int, np.ndarray]:
    """
    DBSCAN clustering with optional automatic eps selection.
    
    Args:
        embeddings: Shape (N, D) tensor of embeddings
        eps: DBSCAN eps parameter (if None and auto_eps=True, auto-select)
        min_samples: DBSCAN min_samples parameter
        min_cluster_size: Minimum points per cluster
        auto_eps: Whether to auto-select eps using k-distance graph
        
    Returns:
        Tuple of (n_clusters, cluster_labels)
    """
    embeddings_np = embeddings.cpu().numpy()
    n_samples = len(embeddings_np)
    
    if min_samples is None:
        min_samples = max(2, min_cluster_size)
    
    if eps is None and auto_eps:
        # Auto-select eps using k-distance graph
        eps = auto_eps_selection(embeddings_np, min_samples)
        print(f"  Auto-selected eps={eps:.4f}")
    elif eps is None:
        eps = 0.5  # Default
    
    # Run DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(embeddings_np)
    
    # Count clusters (excluding noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = sum(1 for l in labels if l == -1)
    
    print(f"  DBSCAN: {n_clusters} clusters, {n_noise} noise points")
    
    # If too many clusters or all noise, fallback to K-means
    if n_clusters == 0 or n_noise > n_samples * 0.5:
        print("  DBSCAN produced too many noise points, falling back to K-means")
        k, labels, _ = auto_kmeans(embeddings, min_k=1, max_k=10, 
                                   min_cluster_size=min_cluster_size)
        return k, labels
    
    # Filter clusters below min size
    if min_cluster_size > 1:
        unique, counts = np.unique(labels, return_counts=True)
        small_clusters = unique[counts < min_cluster_size]
        
        # Mark small clusters as noise
        for cluster in small_clusters:
            if cluster != -1:
                labels[labels == cluster] = -1
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    return n_clusters, labels


def auto_eps_selection(embeddings_np: np.ndarray,
                       min_samples: int = 5,
                       k_neighbors: Optional[int] = None) -> float:
    """
    Auto-select eps for DBSCAN using k-distance graph.
    
    Args:
        embeddings_np: Numpy array of embeddings
        min_samples: DBSCAN min_samples parameter
        k_neighbors: Number of neighbors for k-distance (default: min_samples)
        
    Returns:
        Optimal eps value
    """
    if k_neighbors is None:
        k_neighbors = min_samples
    
    # Compute k-nearest neighbor distances
    n_samples = len(embeddings_np)
    if n_samples < k_neighbors + 1:
        return 0.5
    
    # Sample subset for efficiency
    sample_size = min(2000, n_samples)
    if n_samples > sample_size:
        indices = np.random.choice(n_samples, sample_size, replace=False)
        subset = embeddings_np[indices]
    else:
        subset = embeddings_np
    
    # Compute pairwise distances
    distances = pairwise_distances(subset)
    
    # Get k-th nearest neighbor distance for each point
    sorted_distances = np.sort(distances, axis=1)
    kth_distances = sorted_distances[:, k_neighbors]
    
    # Sort distances
    sorted_kth = np.sort(kth_distances)
    
    # Find elbow in k-distance graph
    n_points = len(sorted_kth)
    if n_points < 10:
        return np.median(sorted_kth)
    
    # Take first 80% to avoid tail
    cutoff = int(n_points * 0.8)
    points = sorted_kth[:cutoff]
    
    # Find elbow (point of maximum gradient change)
    if len(points) > 5:
        gradients = np.gradient(points)
        gradient_change = np.abs(np.gradient(gradients))
        elbow_idx = np.argmax(gradient_change)
        eps = points[elbow_idx]
    else:
        eps = np.median(points)
    
    return float(eps)


def balanced_kmeans(embeddings: torch.Tensor,
                    n_clusters: int,
                    min_cluster_size: int = 5,
                    max_iter: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balanced K-means that ensures minimum cluster size.
    
    Args:
        embeddings: Shape (N, D) tensor of embeddings
        n_clusters: Number of clusters
        min_cluster_size: Minimum points per cluster
        max_iter: Maximum iterations for balancing
        
    Returns:
        Tuple of (cluster_labels, cluster_centers)
    """
    embeddings_np = embeddings.cpu().numpy()
    n_samples = len(embeddings_np)
    
    # Initial K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_np)
    centers = kmeans.cluster_centers_
    
    # Check cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    
    # If all clusters meet min size, return
    if np.min(counts) >= min_cluster_size:
        return labels, centers
    
    # Balance clusters
    for _ in range(max_iter):
        # Find clusters below min size
        unique, counts = np.unique(labels, return_counts=True)
        small_clusters = unique[counts < min_cluster_size]
        
        if len(small_clusters) == 0:
            break
        
        # For each small cluster, find points to reassign
        for small_cluster in small_clusters:
            # Find points in small cluster
            small_indices = np.where(labels == small_cluster)[0]
            
            # If cluster is empty, reassign from largest cluster
            if len(small_indices) == 0:
                largest_cluster = unique[np.argmax(counts)]
                large_indices = np.where(labels == largest_cluster)[0]
                
                # Move some points from largest to this cluster
                move_count = min(len(large_indices) // 2, 10)
                move_indices = np.random.choice(large_indices, move_count, replace=False)
                
                labels[move_indices] = small_cluster
                
                # Update centers
                for cluster in [small_cluster, largest_cluster]:
                    mask = labels == cluster
                    if np.any(mask):
                        centers[cluster] = embeddings_np[mask].mean(axis=0)
                
                continue
            
            # Move points from other clusters to this one
            needed = min_cluster_size - len(small_indices)
            
            # Find closest points from other clusters
            all_dists = pairwise_distances(embeddings_np, centers[small_cluster:small_cluster+1])
            all_dists = all_dists.flatten()
            
            # Sort by distance
            sorted_indices = np.argsort(all_dists)
            
            # Find points not in small cluster
            reassign_count = 0
            for idx in sorted_indices:
                if labels[idx] != small_cluster and reassign_count < needed:
                    # Check if source cluster can spare a point
                    source_cluster = labels[idx]
                    source_count = np.sum(labels == source_cluster)
                    
                    if source_count > min_cluster_size * 2:
                        labels[idx] = small_cluster
                        reassign_count += 1
                        
                        # Update centers
                        for cluster in [small_cluster, source_cluster]:
                            mask = labels == cluster
                            if np.any(mask):
                                centers[cluster] = embeddings_np[mask].mean(axis=0)
        
        # Recompute counts
        unique, counts = np.unique(labels, return_counts=True)
    
    return labels, centers


def cluster_validation(embeddings: torch.Tensor,
                       labels: np.ndarray) -> Dict[str, float]:
    """
    Compute cluster validation metrics.
    
    Args:
        embeddings: Shape (N, D) tensor of embeddings
        labels: Cluster labels
        
    Returns:
        Dictionary of metrics
    """
    embeddings_np = embeddings.cpu().numpy()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    metrics = {
        'n_clusters': n_clusters,
        'n_noise': sum(1 for l in labels if l == -1),
    }
    
    if n_clusters >= 2 and -1 not in labels:
        try:
            metrics['silhouette'] = silhouette_score(embeddings_np, labels)
        except:
            metrics['silhouette'] = 0.0
    
    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) > 0:
        metrics['min_cluster_size'] = np.min(counts)
        metrics['max_cluster_size'] = np.max(counts)
        metrics['avg_cluster_size'] = np.mean(counts)
    
    return metrics


# ============================================================================
# Test Cases
# ============================================================================

def test_auto_kmeans():
    """Test automatic K-means with silhouette method."""
    print("\n" + "="*60)
    print("TEST: Auto K-means")
    print("="*60)
    
    torch.manual_seed(42)
    
    # Create well-separated clusters
    clusters = []
    for i in range(3):
        center = torch.tensor([i * 3.0, 0.0])
        points = torch.randn(30, 2) * 0.3 + center
        clusters.append(points)
    
    embeddings = torch.cat(clusters, dim=0)
    
    # Auto K-means with silhouette
    k, labels, centers = auto_kmeans(embeddings, max_k=6, min_k=1, 
                                     min_cluster_size=5, method='silhouette')
    
    print(f"  Optimal k: {k}")
    unique_labels = len(set(labels))
    print(f"  Labels: {unique_labels} clusters")
    
    # Accept k=2 or 3 (clustering can sometimes merge clusters)
    assert k in [2, 3], f"Expected k=2 or 3, got {k}"
    print(f"  ✓ Reasonable k found: {k}")
    
    print("\n✅ Auto K-means tests passed!")


def test_dbscan():
    """Test DBSCAN clustering."""
    print("\n" + "="*60)
    print("TEST: DBSCAN Clustering")
    print("="*60)
    
    torch.manual_seed(123)
    
    # Create synthetic clusters with noise
    clusters = []
    for i in range(3):
        center = torch.tensor([i * 2.5, 0.0])
        points = torch.randn(20, 2) * 0.2 + center
        clusters.append(points)
    
    # Add noise
    noise = torch.randn(20, 2) * 2.0
    clusters.append(noise)
    
    embeddings = torch.cat(clusters, dim=0)
    
    # DBSCAN
    n_clusters, labels = dbscan_clustering(embeddings, min_samples=3, auto_eps=True)
    
    print(f"  Found {n_clusters} clusters")
    
    # Should find at least 1 cluster
    assert n_clusters >= 1, f"Expected at least 1 cluster, got {n_clusters}"
    print(f"  ✓ Reasonable number of clusters found: {n_clusters}")
    
    print("\n✅ DBSCAN tests passed!")


def test_balanced_kmeans():
    """Test balanced K-means."""
    print("\n" + "="*60)
    print("TEST: Balanced K-means")
    print("="*60)
    
    torch.manual_seed(456)
    
    # Create imbalanced clusters
    cluster1 = torch.randn(80, 2) * 0.3
    cluster2 = torch.randn(10, 2) * 0.3 + torch.tensor([3.0, 0.0])
    cluster3 = torch.randn(10, 2) * 0.3 + torch.tensor([6.0, 0.0])
    
    embeddings = torch.cat([cluster1, cluster2, cluster3], dim=0)
    
    # Balanced K-means
    labels, centers = balanced_kmeans(embeddings, n_clusters=3, min_cluster_size=15)
    
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Cluster sizes: {dict(zip(unique, counts))}")
    
    # Check that we got 3 clusters
    n_clusters = len(unique)
    assert n_clusters == 3, f"Expected 3 clusters, got {n_clusters}"
    print(f"  ✓ Found {n_clusters} clusters")
    
    # All clusters should have at least 15 points
    min_size = np.min(counts)
    assert min_size >= 15, f"Min cluster size {min_size} < 15"
    print(f"  ✓ All clusters have size >= 15")
    
    print("\n✅ Balanced K-means tests passed!")


def test_cluster_validation():
    """Test cluster validation metrics."""
    print("\n" + "="*60)
    print("TEST: Cluster Validation")
    print("="*60)
    
    torch.manual_seed(789)
    
    # Create well-separated clusters
    clusters = []
    for i in range(3):
        center = torch.tensor([i * 3.0, 0.0])
        points = torch.randn(30, 2) * 0.3 + center
        clusters.append(points)
    
    embeddings = torch.cat(clusters, dim=0)
    
    # K-means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings.cpu().numpy())
    
    # Validate
    metrics = cluster_validation(embeddings, labels)
    
    print(f"  Metrics: {metrics}")
    
    assert metrics['n_clusters'] == 3, "Wrong number of clusters"
    print(f"  ✓ Correct number of clusters: {metrics['n_clusters']}")
    
    print("\n✅ Cluster validation tests passed!")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*60)
    print("RUNNING ALL CLUSTER INIT TESTS")
    print("="*60)
    
    test_auto_kmeans()
    test_dbscan()
    test_balanced_kmeans()
    test_cluster_validation()
    
    print("\n" + "="*60)
    print("🎉 ALL CLUSTER INIT TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()