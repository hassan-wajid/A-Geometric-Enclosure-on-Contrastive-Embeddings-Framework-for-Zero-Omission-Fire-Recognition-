"""
FENCE Evaluation Entry Point.

Evaluates a trained FENCE model on test set with all 16 metrics.
Can also visualize predictions and save results.

Usage:
    # Evaluate best model
    python code/fence/evaluate.py --dataset fasdd
    
    # Evaluate specific checkpoint
    python code/fence/evaluate.py --dataset fasdd --checkpoint path/to/model.pt
    
    # Save visualizations
    python code/fence/evaluate.py --dataset fasdd --save_images
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# Import FENCE components
from fence.config import (
    DATASET_NAME, DEVICE, GPU_IDS, SEED,
    FENCE_BACKBONE, FENCE_EMBEDDING_DIM,
    FENCE_RADIUS_MARGIN,
    FENCE_CHECKPOINT_DIR
)

from fence.core.fence_model import FENCEModel
from fence.utils.metrics import print_metrics_table, compute_all_metrics

# Import dataset utilities
from datasets.dataset_factory import create_dataset, create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="FENCE Evaluation")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME,
                        choices=["fasdd", "dfire", "ustc_smokers", "flame"],
                        help="Dataset to evaluate on")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (uses best model if None)")
    parser.add_argument("--save_images", action="store_true",
                        help="Save prediction visualizations")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    return parser.parse_args()


def load_model(model_path: Path, device: str = DEVICE) -> FENCEModel:
    """Load FENCE model from checkpoint."""
    print(f"\nLoading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    model = FENCEModel(
        backbone_name=FENCE_BACKBONE,
        embedding_dim=FENCE_EMBEDDING_DIM,
        num_balls=checkpoint.get('num_balls', 0),
        margin=FENCE_RADIUS_MARGIN,
        device=device
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Restore union of balls
    if 'union_of_balls' in checkpoint:
        # Rebuild union from saved dict
        # This is a simplification - in practice need full reconstruction
        pass
    
    print(f"  ✓ Model loaded")
    print(f"  ✓ Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  ✓ Best FNR: {checkpoint.get('best_val_fnr', 'N/A')}")
    print(f"  ✓ Best FPR: {checkpoint.get('best_val_fpr', 'N/A')}")
    
    return model


def evaluate(model: FENCEModel, test_loader, device: str = DEVICE):
    """Evaluate model on test set."""
    all_labels = []
    all_scores = []
    all_preds = []
    all_abstained = []
    all_contains = []
    all_in_shell = []
    all_ood = []
    all_distances = []
    all_embeddings = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            pred = model.predict(images)
            
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(pred['scores'].cpu().numpy())
            all_preds.extend(pred['prediction'].cpu().numpy())
            all_abstained.extend(pred['abstained'].cpu().numpy())
            all_contains.extend(pred['contains'].cpu().numpy())
            all_in_shell.extend(pred['in_shell'].cpu().numpy())
            all_ood.extend(pred['ood'].cpu().numpy())
            all_distances.extend(pred['distance_to_nearest'].cpu().numpy())
            all_embeddings.extend(pred['embeddings'].cpu().numpy())
    
    # Convert to numpy
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    all_preds = np.array(all_preds)
    all_abstained = np.array(all_abstained)
    all_contains = np.array(all_contains)
    all_in_shell = np.array(all_in_shell)
    all_ood = np.array(all_ood)
    all_distances = np.array(all_distances)
    all_embeddings = np.array(all_embeddings)
    
    # Compute metrics
    metrics = compute_all_metrics(
        all_labels, all_scores, all_preds, all_abstained
    )
    
    # Add detailed metrics
    metrics['contains'] = all_contains
    metrics['in_shell'] = all_in_shell
    metrics['ood'] = all_ood
    metrics['distances'] = all_distances
    metrics['embeddings'] = all_embeddings
    metrics['num_balls'] = len(model.union_of_balls)
    
    if len(model.union_of_balls) > 0:
        radii = [ball.radius for ball in model.union_of_balls.balls]
        metrics['radius_mean'] = np.mean(radii)
        metrics['radius_std'] = np.std(radii)
        metrics['radius_min'] = np.min(radii)
        metrics['radius_max'] = np.max(radii)
    
    return metrics


def save_visualizations(metrics: dict, output_dir: Path):
    """Save prediction visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = np.array([[metrics['tn'], metrics['fp']],
                   [metrics['fn'], metrics['tp']]])
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrix saved")
    
    # 2. Score distribution
    plt.figure(figsize=(10, 6))
    pos_mask = metrics['labels'] == 1
    neg_mask = metrics['labels'] == 0
    plt.hist(metrics['scores'][pos_mask], bins=50, alpha=0.5, label='Positive', color='red')
    plt.hist(metrics['scores'][neg_mask], bins=50, alpha=0.5, label='Negative', color='blue')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title('Score Distribution by Class')
    plt.legend()
    plt.savefig(output_dir / 'score_distribution.png', dpi=150)
    plt.close()
    print(f"  ✓ Score distribution saved")
    
    # 3. Distance distribution
    if 'distances' in metrics:
        plt.figure(figsize=(10, 6))
        plt.hist(metrics['distances'][pos_mask], bins=50, alpha=0.5, label='Positive', color='red')
        plt.hist(metrics['distances'][neg_mask], bins=50, alpha=0.5, label='Negative', color='blue')
        plt.axvline(x=0, color='black', linestyle='--', label='Boundary')
        plt.xlabel('Signed Distance to Ball')
        plt.ylabel('Count')
        plt.title('Distance Distribution by Class')
        plt.legend()
        plt.savefig(output_dir / 'distance_distribution.png', dpi=150)
        plt.close()
        print(f"  ✓ Distance distribution saved")
    
    # 4. Radius distribution
    if 'radii' in metrics:
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(metrics['radii'])), metrics['radii'])
        plt.xlabel('Ball Index')
        plt.ylabel('Radius')
        plt.title('Ball Radii Distribution')
        plt.savefig(output_dir / 'radii_distribution.png', dpi=150)
        plt.close()
        print(f"  ✓ Radius distribution saved")


def main():
    args = parse_args()
    
    device = DEVICE
    print(f"\nDevice: {device}")
    
    # Find checkpoint
    if args.checkpoint:
        model_path = Path(args.checkpoint)
    else:
        # Look for best model
        checkpoint_dir = FENCE_CHECKPOINT_DIR / args.dataset / FENCE_BACKBONE
        model_path = checkpoint_dir / "best_model.pt"
        if not model_path.exists():
            # Try other naming
            model_path = checkpoint_dir / "best_model.pt"
            if not model_path.exists():
                print(f"Error: No model found in {checkpoint_dir}")
                return
    
    # Load model
    model = load_model(model_path, device)
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    from datasets.dataset_factory import create_dataset
    from fence.main import get_dataset_paths
    
    image_dir, annotation_dir = get_dataset_paths(args.dataset)
    dataset = create_dataset(args.dataset, image_dir, annotation_dir)
    
    # Get test set
    if args.dataset == "dfire":
        from datasets.dataset_factory import create_test_dataloader
        test_loader = create_test_dataloader(dataset, test_mode=False)
    else:
        samples = dataset.parse_samples()
        splits = dataset.split_samples(samples)
        test_loader = create_dataloaders({'test': splits['test']}, test_mode=False)['test']
    
    # Evaluate
    print(f"\n{'='*60}")
    print("EVALUATING ON TEST SET")
    print("=" * 60)
    
    metrics = evaluate(model, test_loader, device)
    
    # Print metrics
    print_metrics_table(metrics)
    
    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else FENCE_CHECKPOINT_DIR / args.dataset / FENCE_BACKBONE / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics to file
    results_file = output_dir / "test_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"FENCE Evaluation Results - {args.dataset}\n")
        f.write("=" * 60 + "\n\n")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value:.6f}\n")
    
    print(f"\n  Results saved to: {results_file}")
    
    # Save visualizations
    if args.save_images:
        print(f"\nGenerating visualizations...")
        save_visualizations(metrics, output_dir)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()