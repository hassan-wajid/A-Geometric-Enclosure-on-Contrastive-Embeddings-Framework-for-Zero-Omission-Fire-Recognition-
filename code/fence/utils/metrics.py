"""
Comprehensive metrics for FENCE evaluation.

Implements all metrics required for the FENCE framework:
- Primary: FNR, FPR, FPR@100%Recall
- Detection: AUGRC, Recall, F2-Score
- Abstention: Abstention rate, Risk-Coverage curve, FNR-vs-Abstention
- OOD: OOD AUROC, FPR95
- Geometric: Number of balls, Radius distribution, Separation margin, Slack histogram

Usage:
    python code/fence/utils/metrics.py  # Runs test cases
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
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, auc
)
from scipy.stats import gaussian_kde


def compute_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard confusion matrix metrics.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Predicted labels (0/1)
        
    Returns:
        Dictionary with tn, fp, fn, tp, accuracy, precision, recall, f1, fpr, fnr
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    total_pos = tp + fn
    total_neg = tn + fp
    total = total_pos + total_neg
    
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_pos if total_pos > 0 else 0.0  # = 1 - FNR
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / total_neg if total_neg > 0 else 0.0
    fnr = fn / total_pos if total_pos > 0 else 0.0
    
    return {
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'fnr': fnr,
        'total_pos': total_pos,
        'total_neg': total_neg,
        'total': total
    }


def compute_f2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute F2 score (recall-weighted F-score).
    
    F2 = (1 + 2^2) * (precision * recall) / (2^2 * precision + recall)
    
    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Predicted labels (0/1)
        
    Returns:
        F2 score
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    beta = 2.0
    beta2 = beta ** 2
    
    if (beta2 * precision + recall) == 0:
        return 0.0
    
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def compute_fpr_at_recall(y_true: np.ndarray, y_scores: np.ndarray, 
                          target_recall: float = 1.0) -> float:
    """
    Compute FPR at a target recall (e.g., 100% recall / FNR=0).
    
    Finds the threshold that achieves the target recall, then computes FPR.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_scores: Model scores (higher = more positive)
        target_recall: Desired recall (default: 1.0 = zero omission)
        
    Returns:
        FPR at the target recall, or -1 if impossible
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Sort by score descending
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_scores = y_scores[sorted_indices]
    sorted_labels = y_true[sorted_indices]
    
    # Find positives
    pos_indices = np.where(sorted_labels == 1)[0]
    n_pos = len(pos_indices)
    
    if n_pos == 0:
        return -1.0  # No positives to evaluate
    
    # Number of positives needed to achieve target recall
    n_pos_needed = int(np.ceil(target_recall * n_pos))
    
    if n_pos_needed > n_pos:
        return -1.0  # Impossible
    
    # Get the score at the nth positive
    threshold_idx = pos_indices[n_pos_needed - 1]
    threshold = sorted_scores[threshold_idx]
    
    # Compute predictions at this threshold
    y_pred = (y_scores >= threshold).astype(int)
    
    # Compute metrics
    metrics = compute_confusion_metrics(y_true, y_pred)
    
    return metrics['fpr']


def compute_risk_coverage_curve(y_true: np.ndarray, y_scores: np.ndarray,
                                abstention_mask: Optional[np.ndarray] = None,
                                n_points: int = 100) -> Dict[str, np.ndarray]:
    """
    Compute risk-coverage curve for selective classification.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_scores: Model scores (higher = more positive)
        abstention_mask: Boolean mask indicating abstained samples
        n_points: Number of points for the curve
        
    Returns:
        Dictionary with 'coverage', 'risk', 'accuracies'
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # If abstention mask provided, only consider non-abstained samples
    if abstention_mask is not None:
        mask = ~abstention_mask
        y_true = y_true[mask]
        y_scores = y_scores[mask]
    
    n = len(y_true)
    if n == 0:
        return {'coverage': np.array([0.0, 1.0]), 'risk': np.array([1.0, 0.0])}
    
    # Sort by score descending
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_labels = y_true[sorted_indices]
    
    # Compute cumulative metrics
    coverage = np.linspace(0.1, 1.0, n_points)
    risks = []
    accuracies = []
    
    for cov in coverage:
        n_select = int(np.ceil(cov * n))
        if n_select == 0:
            selected_labels = np.array([])
        else:
            selected_labels = sorted_labels[:n_select]
        
        if len(selected_labels) == 0:
            risk = 1.0
            acc = 0.0
        else:
            risk = 1.0 - np.mean(selected_labels)  # Proportion of negatives selected
            acc = np.mean(selected_labels == 1)    # Accuracy on selected
        
        risks.append(risk)
        accuracies.append(acc)
    
    return {
        'coverage': coverage,
        'risk': np.array(risks),
        'accuracy': np.array(accuracies)
    }


def compute_augrc(y_true: np.ndarray, y_scores: np.ndarray,
                  abstention_mask: Optional[np.ndarray] = None) -> float:
    """
    Compute Area Under the Generalized Risk-Coverage curve (AUGRC).
    
    AUGRC measures the average risk of undetected failures.
    Lower is better.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_scores: Model scores (higher = more positive)
        abstention_mask: Boolean mask indicating abstained samples
        
    Returns:
        AUGRC value
    """
    curve = compute_risk_coverage_curve(y_true, y_scores, abstention_mask)
    
    # Integrate risk over coverage
    coverage = curve['coverage']
    risk = curve['risk']
    
    # Trapezoidal integration
    augrc = np.trapz(risk, coverage)
    
    return augrc


def compute_ood_metrics(indist_scores: np.ndarray, ood_scores: np.ndarray) -> Dict[str, float]:
    """
    Compute OOD detection metrics.
    
    Args:
        indist_scores: Scores for in-distribution samples (higher = more ID)
        ood_scores: Scores for out-of-distribution samples (lower = more OOD)
        
    Returns:
        Dictionary with AUROC and FPR95
    """
    # Combine labels: 1 for ID, 0 for OOD
    y_true = np.concatenate([np.ones_like(indist_scores), np.zeros_like(ood_scores)])
    y_scores = np.concatenate([indist_scores, ood_scores])
    
    # AUROC
    auroc = roc_auc_score(y_true, y_scores)
    
    # FPR95: False Positive Rate at 95% True Positive Rate
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Find threshold closest to 95% TPR
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx] if idx < len(fpr) else 1.0
    
    return {
        'ood_auroc': auroc,
        'ood_fpr95': fpr95
    }


def compute_abstention_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                abstention_mask: np.ndarray) -> Dict[str, float]:
    """
    Compute abstention-related metrics.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Predicted labels (0/1) for non-abstained samples
        abstention_mask: Boolean mask indicating abstained samples
        
    Returns:
        Dictionary with abstention rate and composition
    """
    n = len(y_true)
    n_abstained = np.sum(abstention_mask)
    abstention_rate = n_abstained / n if n > 0 else 0.0
    
    # Composition of abstained cases
    abstained_true = y_true[abstention_mask]
    n_abstained_pos = np.sum(abstained_true == 1)
    n_abstained_neg = np.sum(abstained_true == 0)
    
    # For non-abstained, compute metrics
    non_abstained_mask = ~abstention_mask
    if np.sum(non_abstained_mask) > 0:
        y_true_non = y_true[non_abstained_mask]
        y_pred_non = y_pred[non_abstained_mask]
        metrics_non = compute_confusion_metrics(y_true_non, y_pred_non)
    else:
        metrics_non = {'fnr': 1.0, 'fpr': 0.0, 'accuracy': 0.0}
    
    return {
        'abstention_rate': abstention_rate,
        'abstained_pos': n_abstained_pos,
        'abstained_neg': n_abstained_neg,
        'abstained_pos_ratio': n_abstained_pos / n_abstained if n_abstained > 0 else 0.0,
        'non_abstained_fnr': metrics_non['fnr'],
        'non_abstained_fpr': metrics_non['fpr'],
        'non_abstained_accuracy': metrics_non['accuracy']
    }


def compute_geometry_metrics(union_of_balls, embeddings: torch.Tensor,
                             labels: torch.Tensor) -> Dict[str, Any]:
    """
    Compute geometric diagnostics for the enclosure.
    
    Args:
        union_of_balls: UnionOfBalls object
        embeddings: Feature embeddings (N, D)
        labels: Labels (0/1)
        
    Returns:
        Dictionary with geometric metrics
    """
    metrics = {}
    
    # Number of balls
    metrics['num_balls'] = len(union_of_balls)
    
    # Radius distribution
    if len(union_of_balls) > 0:
        radii = [ball.radius for ball in union_of_balls.balls]
        metrics['radii'] = radii
        metrics['radius_mean'] = np.mean(radii)
        metrics['radius_std'] = np.std(radii)
        metrics['radius_min'] = np.min(radii)
        metrics['radius_max'] = np.max(radii)
    
    # Separation margin
    separation_margin = union_of_balls.get_min_separation_margin(embeddings, labels)
    metrics['separation_margin'] = separation_margin
    
    # Margin distribution per ball
    margins = []
    if len(union_of_balls) > 0:
        for ball in union_of_balls.balls:
            # Distance from center to all points
            dists = ball.distance(embeddings)
            
            # Positives in this ball
            pos_mask = (labels == 1) & (dists <= ball.radius)
            if pos_mask.any():
                R_plus = dists[pos_mask].max().item()
            else:
                R_plus = 0.0
            
            # Nearest negative
            neg_mask = (labels == 0)
            if neg_mask.any():
                neg_dists = dists[neg_mask]
                if len(neg_dists) > 0:
                    R_minus = neg_dists.min().item()
                    margins.append(R_minus - R_plus)
        
        metrics['ball_margins'] = margins if margins else [0.0]
        metrics['min_ball_margin'] = np.min(margins) if margins else 0.0
    
    # Certified radius (assuming L=1 for feature space)
    metrics['certified_radius_feature'] = union_of_balls.margin
    
    return metrics


def compute_all_metrics(y_true: np.ndarray, y_scores: np.ndarray,
                        y_pred: np.ndarray, abstention_mask: np.ndarray,
                        union_of_balls=None, embeddings=None, labels=None,
                        indist_scores=None, ood_scores=None) -> Dict[str, Any]:
    """
    Compute all FENCE metrics in one call.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_scores: Model scores (higher = more positive)
        y_pred: Predicted labels (0/1)
        abstention_mask: Boolean mask indicating abstained samples
        union_of_balls: UnionOfBalls object (for geometry metrics)
        embeddings: Feature embeddings (for geometry metrics)
        labels: Labels (for geometry metrics)
        indist_scores: In-distribution scores (for OOD metrics)
        ood_scores: Out-of-distribution scores (for OOD metrics)
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Primary metrics
    cm = compute_confusion_metrics(y_true, y_pred)
    metrics['tn'] = cm['tn']
    metrics['fp'] = cm['fp']
    metrics['fn'] = cm['fn']
    metrics['tp'] = cm['tp']
    metrics['accuracy'] = cm['accuracy']
    metrics['precision'] = cm['precision']
    metrics['recall'] = cm['recall']
    metrics['f1'] = cm['f1']
    metrics['fpr'] = cm['fpr']
    metrics['fnr'] = cm['fnr']
    
    # F2 score
    metrics['f2'] = compute_f2_score(y_true, y_pred)
    
    # FPR at 100% recall (zero omission)
    fpr_at_recall = compute_fpr_at_recall(y_true, y_scores, target_recall=1.0)
    metrics['fpr_at_100_recall'] = fpr_at_recall
    
    # AUGRC (risk-coverage curve)
    augrc = compute_augrc(y_true, y_scores, abstention_mask)
    metrics['augrc'] = augrc
    
    # Risk-coverage curve (for plotting)
    risk_coverage = compute_risk_coverage_curve(y_true, y_scores, abstention_mask)
    metrics['risk_coverage_curve'] = risk_coverage
    
    # Abstention metrics
    abst_metrics = compute_abstention_metrics(y_true, y_pred, abstention_mask)
    metrics['abstention_rate'] = abst_metrics['abstention_rate']
    metrics['abstained_pos'] = abst_metrics['abstained_pos']
    metrics['abstained_neg'] = abst_metrics['abstained_neg']
    metrics['abstained_pos_ratio'] = abst_metrics['abstained_pos_ratio']
    metrics['non_abstained_fnr'] = abst_metrics['non_abstained_fnr']
    metrics['non_abstained_fpr'] = abst_metrics['non_abstained_fpr']
    
    # OOD metrics
    if indist_scores is not None and ood_scores is not None:
        ood_metrics = compute_ood_metrics(indist_scores, ood_scores)
        metrics['ood_auroc'] = ood_metrics['ood_auroc']
        metrics['ood_fpr95'] = ood_metrics['ood_fpr95']
    
    # Geometric metrics
    if union_of_balls is not None and embeddings is not None and labels is not None:
        geo_metrics = compute_geometry_metrics(union_of_balls, embeddings, labels)
        metrics['num_balls'] = geo_metrics['num_balls']
        metrics['radii'] = geo_metrics.get('radii', [])
        metrics['radius_mean'] = geo_metrics.get('radius_mean', 0.0)
        metrics['radius_std'] = geo_metrics.get('radius_std', 0.0)
        metrics['separation_margin'] = geo_metrics.get('separation_margin', 0.0)
        metrics['certified_radius_feature'] = geo_metrics.get('certified_radius_feature', 0.0)
    
    return metrics


def print_metrics_table(metrics: Dict[str, Any]) -> None:
    """
    Print metrics in a formatted table.
    
    Args:
        metrics: Dictionary of metrics from compute_all_metrics
    """
    print("\n" + "="*60)
    print("FENCE EVALUATION METRICS")
    print("="*60)
    
    print("\n--- PRIMARY METRICS ---")
    print(f"  FNR (False Negative Rate):  {metrics.get('fnr', 0.0):.6f}")
    print(f"  FPR (False Positive Rate):  {metrics.get('fpr', 0.0):.6f}")
    print(f"  FPR@100% Recall:            {metrics.get('fpr_at_100_recall', 0.0):.6f}")
    
    print("\n--- DETECTION METRICS ---")
    print(f"  Recall (1-FNR):             {metrics.get('recall', 0.0):.6f}")
    print(f"  Precision:                  {metrics.get('precision', 0.0):.6f}")
    print(f"  F1 Score:                   {metrics.get('f1', 0.0):.6f}")
    print(f"  F2 Score:                   {metrics.get('f2', 0.0):.6f}")
    print(f"  Accuracy:                   {metrics.get('accuracy', 0.0):.6f}")
    print(f"  AUGRC:                      {metrics.get('augrc', 0.0):.6f}")
    
    print("\n--- ABSTRACTION METRICS ---")
    print(f"  Abstention Rate:            {metrics.get('abstention_rate', 0.0):.6f}")
    print(f"  Abstained Pos:              {metrics.get('abstained_pos', 0)}")
    print(f"  Abstained Neg:              {metrics.get('abstained_neg', 0)}")
    print(f"  Pos Ratio in Abstained:     {metrics.get('abstained_pos_ratio', 0.0):.6f}")
    print(f"  Non-Abstained FNR:          {metrics.get('non_abstained_fnr', 0.0):.6f}")
    print(f"  Non-Abstained FPR:          {metrics.get('non_abstained_fpr', 0.0):.6f}")
    
    if 'ood_auroc' in metrics:
        print("\n--- OOD METRICS ---")
        print(f"  OOD AUROC:                 {metrics.get('ood_auroc', 0.0):.6f}")
        print(f"  OOD FPR95:                 {metrics.get('ood_fpr95', 0.0):.6f}")
    
    if 'num_balls' in metrics:
        print("\n--- GEOMETRY METRICS ---")
        print(f"  Number of Balls:           {metrics.get('num_balls', 0)}")
        print(f"  Mean Radius:               {metrics.get('radius_mean', 0.0):.4f}")
        print(f"  Std Radius:                {metrics.get('radius_std', 0.0):.4f}")
        print(f"  Separation Margin:         {metrics.get('separation_margin', 0.0):.4f}")
        print(f"  Certified Radius (feat):   {metrics.get('certified_radius_feature', 0.0):.4f}")
    
    print("\n--- CONFUSION MATRIX ---")
    print(f"  TN = {metrics.get('tn', 0):>6}  |  FP = {metrics.get('fp', 0):>6}")
    print(f"  FN = {metrics.get('fn', 0):>6}  |  TP = {metrics.get('tp', 0):>6}")
    
    # Zero omission status
    fnr = metrics.get('fnr', 1.0)
    fpr = metrics.get('fpr', 1.0)
    if fnr == 0.0:
        print(f"\n  ✅ ZERO OMISSION ACHIEVED (FNR = 0.000000)")
        if fpr < 0.1:
            print(f"  ✅ OPERATIONALLY USABLE (FPR < 10%)")
        else:
            print(f"  ⚠️  High FPR: {fpr*100:.1f}%")
    else:
        print(f"\n  ❌ Zero omission NOT achieved (FNR = {fnr:.6f})")
    
    print("="*60)


# ============================================================================
# Test Cases
# ============================================================================

def test_primary_metrics():
    """Test primary metrics (FNR, FPR, FPR@100%Recall)."""
    print("\n" + "="*60)
    print("TEST: Primary Metrics")
    print("="*60)
    
    # Create test data
    np.random.seed(42)
    n = 1000
    y_true = np.random.randint(0, 2, n)
    y_scores = np.random.rand(n)
    
    # Positive scores should be higher
    y_scores[y_true == 1] += 0.3
    
    # Compute predictions at threshold 0.5
    y_pred = (y_scores >= 0.5).astype(int)
    
    # Compute metrics
    cm = compute_confusion_metrics(y_true, y_pred)
    
    print(f"  Confusion matrix:")
    print(f"    TN={cm['tn']}, FP={cm['fp']}")
    print(f"    FN={cm['fn']}, TP={cm['tp']}")
    print(f"  FNR={cm['fnr']:.4f}, FPR={cm['fpr']:.4f}")
    
    # Test FPR at 100% recall
    fpr_at_recall = compute_fpr_at_recall(y_true, y_scores, target_recall=1.0)
    print(f"  FPR at 100% recall: {fpr_at_recall:.4f}")
    
    assert 0 <= cm['fnr'] <= 1, "FNR out of range"
    assert 0 <= cm['fpr'] <= 1, "FPR out of range"
    assert 0 <= fpr_at_recall <= 1, "FPR at recall out of range"
    
    print("\n✅ Primary metrics tests passed!")


def test_f2_score():
    """Test F2 score computation."""
    print("\n" + "="*60)
    print("TEST: F2 Score")
    print("="*60)
    
    # Case 1: Perfect
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    f2 = compute_f2_score(y_true, y_pred)
    print(f"  Perfect: F2={f2:.4f} (expected 1.0)")
    assert abs(f2 - 1.0) < 1e-6, "F2 should be 1.0 for perfect"
    
    # Case 2: All positive
    y_pred = np.array([1, 1, 1, 1])
    f2 = compute_f2_score(y_true, y_pred)
    print(f"  All positive: F2={f2:.4f}")
    
    # Case 3: All negative
    y_pred = np.array([0, 0, 0, 0])
    f2 = compute_f2_score(y_true, y_pred)
    print(f"  All negative: F2={f2:.4f}")
    
    # F2 should be recall-weighted
    assert 0 <= f2 <= 1, "F2 out of range"
    
    print("\n✅ F2 score tests passed!")


def test_augrc():
    """Test AUGRC computation."""
    print("\n" + "="*60)
    print("TEST: AUGRC")
    print("="*60)
    
    np.random.seed(42)
    n = 500
    y_true = np.random.randint(0, 2, n)
    y_scores = np.random.rand(n)
    y_scores[y_true == 1] += 0.5
    
    # No abstention
    abstention_mask = np.zeros(n, dtype=bool)
    
    augrc = compute_augrc(y_true, y_scores, abstention_mask)
    print(f"  AUGRC (no abstention): {augrc:.4f}")
    
    # With abstention (random)
    abstention_mask = np.random.rand(n) > 0.8
    augrc_abstain = compute_augrc(y_true, y_scores, abstention_mask)
    print(f"  AUGRC (with abstention): {augrc_abstain:.4f}")
    
    # AUGRC should be between 0 and 1
    assert 0 <= augrc <= 1, "AUGRC out of range"
    
    print("\n✅ AUGRC tests passed!")


def test_ood_metrics():
    """Test OOD metrics."""
    print("\n" + "="*60)
    print("TEST: OOD Metrics")
    print("="*60)
    
    np.random.seed(42)
    
    # ID scores: high
    indist_scores = np.random.normal(0.8, 0.1, 1000)
    indist_scores = np.clip(indist_scores, 0, 1)
    
    # OOD scores: low
    ood_scores = np.random.normal(0.2, 0.1, 1000)
    ood_scores = np.clip(ood_scores, 0, 1)
    
    ood_metrics = compute_ood_metrics(indist_scores, ood_scores)
    print(f"  OOD AUROC: {ood_metrics['ood_auroc']:.4f}")
    print(f"  OOD FPR95: {ood_metrics['ood_fpr95']:.4f}")
    
    # Should be good separation
    assert ood_metrics['ood_auroc'] > 0.8, "AUROC should be > 0.8 for well-separated data"
    
    print("\n✅ OOD metrics tests passed!")


def test_abstention_metrics():
    """Test abstention metrics."""
    print("\n" + "="*60)
    print("TEST: Abstention Metrics")
    print("="*60)
    
    np.random.seed(42)
    n = 1000
    y_true = np.random.randint(0, 2, n)
    y_pred = np.random.randint(0, 2, n)
    abstention_mask = np.random.rand(n) > 0.8  # 20% abstention
    
    abst_metrics = compute_abstention_metrics(y_true, y_pred, abstention_mask)
    
    print(f"  Abstention rate: {abst_metrics['abstention_rate']:.4f}")
    print(f"  Abstained positives: {abst_metrics['abstained_pos']}")
    print(f"  Abstained negatives: {abst_metrics['abstained_neg']}")
    print(f"  Non-abstained FNR: {abst_metrics['non_abstained_fnr']:.4f}")
    
    assert 0 <= abst_metrics['abstention_rate'] <= 1, "Abstention rate out of range"
    
    print("\n✅ Abstention metrics tests passed!")


def test_full_metrics():
    """Test the full metrics pipeline."""
    print("\n" + "="*60)
    print("TEST: Full Metrics Pipeline")
    print("="*60)
    
    np.random.seed(42)
    n = 1000
    y_true = np.random.randint(0, 2, n)
    y_scores = np.random.rand(n)
    y_scores[y_true == 1] += 0.3
    y_pred = (y_scores >= 0.5).astype(int)
    abstention_mask = np.random.rand(n) > 0.85
    
    # OOD scores
    indist_scores = np.random.normal(0.8, 0.1, 1000)
    ood_scores = np.random.normal(0.2, 0.1, 1000)
    indist_scores = np.clip(indist_scores, 0, 1)
    ood_scores = np.clip(ood_scores, 0, 1)
    
    metrics = compute_all_metrics(
        y_true, y_scores, y_pred, abstention_mask,
        indist_scores=indist_scores, ood_scores=ood_scores,
        union_of_balls=None, embeddings=None, labels=None
    )
    
    print_metrics_table(metrics)
    
    print("\n✅ Full metrics pipeline tests passed!")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*60)
    print("RUNNING ALL METRICS TESTS")
    print("="*60)
    
    test_primary_metrics()
    test_f2_score()
    test_augrc()
    test_ood_metrics()
    test_abstention_metrics()
    test_full_metrics()
    
    print("\n" + "="*60)
    print("🎉 ALL METRICS TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()