"""
Test FENCE on all 4 datasets with max 5k samples each.

Runs training and evaluation on:
- FASDD
- D-Fire
- USTC_SmokeRS
- FLAME

Each dataset is limited to 5k samples for quick testing.

Usage:
    python code/fence/test_all_datasets.py
    python code/fence/test_all_datasets.py --epochs 3
    python code/fence/test_all_datasets.py --no_wandb
"""

import os
import sys
from pathlib import Path

# Add parent directory (code/) to path FIRST
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np
from tqdm import tqdm
import random
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Now import everything
from fence.config import (
    DEVICE, GPU_IDS, SEED,
    FENCE_BACKBONE, FENCE_EMBEDDING_DIM,
    FENCE_NUM_CLUSTERS, FENCE_MAX_CLUSTERS, FENCE_MIN_CLUSTER_SIZE,
    FENCE_RADIUS_MARGIN,
    FENCE_LEARNING_RATE, FENCE_WEIGHT_DECAY,
    FENCE_NUM_EPOCHS, FENCE_EVAL_EVERY_N_EPOCHS,
    FENCE_CHECKPOINT_DIR, FENCE_LOG_DIR,
    MSC_TEMPERATURE, MSC_LAMBDA,
    SPECTRAL_NORM_ENABLED
)

from fence.core.fence_model import FENCEModel
from fence.training.feasible_trainer import FeasibleTrainer
from fence.utils.metrics import print_metrics_table, compute_all_metrics

# Import dataset utilities from root
from train_datasets.dataset_factory import create_dataset, create_dataloaders
from sklearn.model_selection import train_test_split


def set_seed(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_dataset_paths(dataset_name: str):
    """Get dataset paths for the specified dataset."""
    base_dir = Path(__file__).parent.parent.parent  # Go up to root
    
    if dataset_name == "fasdd":
        image_dir = base_dir / "dataset" / "FASDD" / "FASDD" / "images"
        annotation_dir = base_dir / "dataset" / "FASDD" / "FASDD" / "annotations" / "VOC" / "Annotations"
    elif dataset_name == "dfire":
        image_dir = base_dir / "dataset" / "D-Fire" / "train" / "images"
        annotation_dir = base_dir / "dataset" / "D-Fire" / "train" / "labels"
    elif dataset_name == "ustc_smokers":
        image_dir = base_dir / "dataset" / "USTC_SmokeRS"
        annotation_dir = base_dir / "dataset" / "USTC_SmokeRS"
    elif dataset_name == "flame":
        image_dir = base_dir / "dataset" / "FLAMEDataset" / "Training" / "Training"
        annotation_dir = base_dir / "dataset" / "FLAMEDataset" / "Training" / "Training"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return image_dir, annotation_dir


def load_dataset(dataset_name: str, image_dir: Path, annotation_dir: Path, 
                 sample_limit: int = 5000):
    """Load dataset with sample limit. No calibration set needed for FENCE."""
    
    print(f"\n{'='*60}")
    print(f"LOADING DATASET: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Create dataset instance
    dataset = create_dataset(dataset_name, image_dir, annotation_dir)
    
    # Parse samples
    if dataset_name == "dfire":
        train_samples = dataset.parse_samples()
        test_samples = dataset.parse_test_samples()
        all_samples = train_samples + test_samples
    elif dataset_name == "ustc_smokers":
        samples = dataset.parse_samples()
        test_samples = dataset.parse_test_samples()
        all_samples = samples + test_samples
    else:
        all_samples = dataset.parse_samples()
        test_samples = []
    
    print(f"  Total samples: {len(all_samples)}")
    pos = sum(1 for s in all_samples if s['label'] == 1)
    neg = len(all_samples) - pos
    print(f"  Positive: {pos}, Negative: {neg}")
    
    # Apply sample limit
    if len(all_samples) > sample_limit:
        random.seed(SEED)
        pos_samples = [s for s in all_samples if s['label'] == 1]
        neg_samples = [s for s in all_samples if s['label'] == 0]
        
        # Sample balanced subset
        pos_limit = min(len(pos_samples), sample_limit // 2)
        neg_limit = min(len(neg_samples), sample_limit - pos_limit)
        
        pos_subset = random.sample(pos_samples, pos_limit) if pos_limit > 0 else []
        neg_subset = random.sample(neg_samples, neg_limit) if neg_limit > 0 else []
        
        all_samples = pos_subset + neg_subset
        random.shuffle(all_samples)
        
        print(f"  ⚠️ Limited to {len(all_samples)} samples")
        print(f"    Positive: {len(pos_subset)}, Negative: {len(neg_subset)}")
    
    # Split into train/val (NO CALIBRATION - FENCE uses constraints during training)
    pos_samples = [s for s in all_samples if s['label'] == 1]
    neg_samples = [s for s in all_samples if s['label'] == 0]
    
    # 80% train, 20% val (no calibration needed)
    pos_train, pos_val = train_test_split(pos_samples, test_size=0.20, random_state=SEED)
    neg_train, neg_val = train_test_split(neg_samples, test_size=0.20, random_state=SEED)
    
    train_samples = pos_train + neg_train
    val_samples = pos_val + neg_val
    
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    
    splits = {
        'train': train_samples,
        'val': val_samples,
        'cal': [],  # Empty - FENCE doesn't need calibration
        'test': []  # We'll use validation for testing in quick run
    }
    
    print(f"\n  Split sizes:")
    for split_name, split_samples in splits.items():
        if split_samples:
            pos_count = sum(1 for s in split_samples if s['label'] == 1)
            print(f"    {split_name}: {len(split_samples)} (pos: {pos_count})")
    
    # Create dataloaders
    dataloaders = create_dataloaders(splits, test_mode=False)
    
    # For datasets with external test, use it
    if dataset_name in ["dfire", "ustc_smokers"] and test_samples:
        from train_datasets.dataset_factory import create_test_dataloader
        test_loader = create_test_dataloader(dataset, test_mode=False)
        print(f"    test: {len(test_loader.dataset)} (external)")
    else:
        test_loader = None
    
    return {
        'train_loader': dataloaders['train'],
        'val_loader': dataloaders['val'],
        'cal_loader': None,  # No calibration needed
        'test_loader': test_loader,
        'splits': splits,
        'dataset': dataset,
        'dataset_name': dataset_name
    }


def train_and_evaluate(dataset_name: str, epochs: int = 3, use_wandb: bool = False):
    """Train and evaluate FENCE on a single dataset."""
    
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING FENCE ON {dataset_name.upper()}")
    print(f"   Epochs: {epochs}")
    print(f"   Backbone: {FENCE_BACKBONE}")
    print(f"   Checkpoints: {FENCE_CHECKPOINT_DIR / dataset_name / FENCE_BACKBONE}")
    print(f"{'='*60}\n")
    
    set_seed(SEED)
    device = DEVICE
    
    # Get paths
    image_dir, annotation_dir = get_dataset_paths(dataset_name)
    
    # Load dataset (max 5k samples)
    data = load_dataset(dataset_name, image_dir, annotation_dir, sample_limit=5000)
    
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    
    # Get positive count
    pos_count = sum(1 for s in data['splits']['train'] if s['label'] == 1)
    print(f"\n  Training positives: {pos_count}")
    
    # Initialize model
    print(f"\n  Initializing FENCE model...")
    model = FENCEModel(
        backbone_name=FENCE_BACKBONE,
        embedding_dim=FENCE_EMBEDDING_DIM,
        num_balls=FENCE_NUM_CLUSTERS,
        max_balls=FENCE_MAX_CLUSTERS,
        min_cluster_size=FENCE_MIN_CLUSTER_SIZE,
        margin=FENCE_RADIUS_MARGIN,
        temperature=MSC_TEMPERATURE,
        lambda_class=MSC_LAMBDA,
        use_mean_shift=True,
        spectral_norm=SPECTRAL_NORM_ENABLED,
        device=device
    )
    
    # Initialize constraints
    model.init_constraints(pos_count)
    
    # Initialize union of balls with training embeddings
    print("\n  Extracting initial embeddings...")
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="  Extracting"):
            images = images.to(device)
            labels = labels.to(device)
            embeddings = model.get_embeddings(images)
            all_embeddings.append(embeddings)
            all_labels.append(labels)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Fit union
    model.fit_union(all_embeddings, all_labels)
    print(f"  Balls initialized: {len(model.union_of_balls)}")
    
    # Store OOD reference
    model.store_ood_reference(all_embeddings, all_labels)
    
    # Setup checkpoint dir
    checkpoint_dir = FENCE_CHECKPOINT_DIR / dataset_name / FENCE_BACKBONE
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Checkpoint directory: {checkpoint_dir}")
    
    # Trainer
    trainer = FeasibleTrainer(
        model=model,
        union_of_balls=model.union_of_balls,
        train_loader=train_loader,
        val_loader=val_loader,
        constraints=model.constraints,
        learning_rate=FENCE_LEARNING_RATE,
        weight_decay=FENCE_WEIGHT_DECAY,
        num_epochs=epochs,
        eval_freq=1,
        device=device,
        use_amp=False,
        save_dir=checkpoint_dir,
        wandb_logger=None
    )
    
    # Train
    print(f"\n  Starting training for {epochs} epochs...")
    start_time = time.time()
    history = trainer.train()
    train_time = time.time() - start_time


    # --- FINALIZE AND VERIFY ---
    print(f"\n  Finalizing model...")
    all_embeddings = []
    all_labels_full = []
    with torch.no_grad():
        for batch_data in train_loader:
            if len(batch_data) == 3:
                images, labels, _ = batch_data
            else:
                images, labels = batch_data
            images = images.to(device)
            outputs = model(images)
            all_embeddings.append(outputs['embeddings'])
            all_labels_full.append(labels.to(device))
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels_full = torch.cat(all_labels_full, dim=0)
    
    model.finalize(all_embeddings, all_labels_full, trainer.multipliers)
    
    print("  Verifying zero FNR on training data...")
    verified = model.verify_zero_fnr(all_embeddings, all_labels_full)
    if not verified:
        print("  ⚠️ WARNING: Zero FNR certificate FAILED!")
    else:
        print("  ✅ Zero FNR certificate PASSED!")
    
    # Final evaluation on validation set
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION - {dataset_name.upper()}")
    print(f"{'='*60}")
    
    all_labels = []
    all_scores = []
    all_preds = []
    all_abstained = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="  Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            pred = model.predict(images)
            
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(pred['scores'].cpu().numpy())
            all_preds.extend(pred['prediction'].cpu().numpy())
            all_abstained.extend(pred['abstained'].cpu().numpy())
    
    # Convert to numpy
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    all_preds = np.array(all_preds)
    all_abstained = np.array(all_abstained)
    
    # Compute metrics
    metrics = compute_all_metrics(
        all_labels, all_scores, all_preds, all_abstained
    )
    
    # Add geometric metrics
    metrics['num_balls'] = len(model.union_of_balls)
    if len(model.union_of_balls) > 0:
        radii = [ball.radius for ball in model.union_of_balls.balls]
        metrics['radius_mean'] = np.mean(radii)
        metrics['radius_std'] = np.std(radii)
        metrics['radius_min'] = np.min(radii)
        metrics['radius_max'] = np.max(radii)
    
    # Print results
    print_metrics_table(metrics)
    
    # Save checkpoint
    best_model_path = checkpoint_dir / "best_model.pt"
    trainer.save_checkpoint("best_model.pt")
    print(f"\n  💾 Best model saved to: {best_model_path}")

    # Also save a summary file with metrics
    summary_path = checkpoint_dir / "test_summary.json"

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    summary = {
        'dataset': dataset_name,
        'backbone': FENCE_BACKBONE,
        'epochs': epochs,
        'train_time': train_time,
        'samples': len(data['splits']['train']),
        'val_samples': len(data['splits']['val']),
        'balls': len(model.union_of_balls),
        'best_val_fnr': float(trainer.best_val_fnr) if trainer.best_val_fnr != float('inf') else None,
        'best_val_fpr': float(trainer.best_val_fpr) if trainer.best_val_fpr != float('inf') else None,
        'test_fnr': float(metrics['fnr']),
        'test_fpr': float(metrics['fpr']),
        'test_f1': float(metrics['f1']),
        'test_recall': float(metrics['recall']),
        'test_precision': float(metrics['precision']),
        'abstention_rate': float(metrics.get('abstention_rate', 0.0)),
        'zero_omission': bool(metrics['fnr'] == 0.0),
        'operational': bool(metrics['fnr'] == 0.0 and metrics['fpr'] < 0.1)
    }

    # Convert to serializable
    summary = convert_to_serializable(summary)

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  📊 Summary saved to: {summary_path}")
    
    # Also evaluate on test set if available
    if test_loader is not None:
        print(f"\n  Also evaluating on external test set...")
        test_labels = []
        test_scores = []
        test_preds = []
        test_abstained = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="  Testing"):
                images = images.to(device)
                labels = labels.to(device)
                
                pred = model.predict(images)
                
                test_labels.extend(labels.cpu().numpy())
                test_scores.extend(pred['scores'].cpu().numpy())
                test_preds.extend(pred['prediction'].cpu().numpy())
                test_abstained.extend(pred['abstained'].cpu().numpy())
        
        test_labels = np.array(test_labels)
        test_scores = np.array(test_scores)
        test_preds = np.array(test_preds)
        test_abstained = np.array(test_abstained)
        
        test_metrics = compute_all_metrics(
            test_labels, test_scores, test_preds, test_abstained
        )
        
        print(f"\n  Test Set Results:")
        print(f"    FNR: {test_metrics['fnr']:.6f}")
        print(f"    FPR: {test_metrics['fpr']:.6f}")
        print(f"    F1: {test_metrics['f1']:.6f}")
        
        # Save test metrics
        test_summary_path = checkpoint_dir / "test_summary_external.json"
        with open(test_summary_path, 'w') as f:
            json.dump(test_metrics, f, indent=2, default=str)
        print(f"  📊 Test metrics saved to: {test_summary_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Test FENCE on all datasets")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs per dataset")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["fasdd", "dfire", "ustc_smokers", "flame"],
                        choices=["fasdd", "dfire", "ustc_smokers", "flame"],
                        help="Datasets to test")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable WandB logging")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🔬 FENCE TESTING ON ALL DATASETS")
    print(f"   Epochs per dataset: {args.epochs}")
    print(f"   Max samples: 5,000 per dataset")
    print(f"   Datasets: {', '.join(args.datasets)}")
    print(f"   Backbone: {FENCE_BACKBONE}")
    print(f"   Checkpoint dir: {FENCE_CHECKPOINT_DIR}")
    print(f"   Device: {DEVICE}")
    print("="*60 + "\n")
    
    all_results = {}
    
    for i, dataset_name in enumerate(args.datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# DATASET {i}/{len(args.datasets)}: {dataset_name.upper()}")
        print(f"{'#'*60}")
        
        try:
            result = train_and_evaluate(dataset_name, epochs=args.epochs)
            all_results[dataset_name] = result
            
            print(f"\n✅ {dataset_name.upper()} COMPLETE")
            print(f"   Best FNR: {result['best_val_fnr']:.6f}")
            print(f"   Best FPR: {result['best_val_fpr']:.6f}")
            print(f"   Test FNR: {result['test_fnr']:.6f}")
            print(f"   Test FPR: {result['test_fpr']:.6f}")
            print(f"   Zero Omission: {'✅ YES' if result['zero_omission'] else '❌ NO'}")
            print(f"   Operational: {'✅ YES' if result['operational'] else '❌ NO'}")
            print(f"   Checkpoint: {FENCE_CHECKPOINT_DIR / dataset_name / FENCE_BACKBONE / 'best_model.pt'}")
            
        except Exception as e:
            print(f"\n❌ {dataset_name.upper()} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset_name] = {'error': str(e)}
    
    # Print summary
    print("\n" + "="*60)
    print("📊 SUMMARY RESULTS")
    print("="*60)
    
    print(f"\n{'Dataset':<15} {'Balls':<8} {'FNR':<12} {'FPR':<12} {'Zero-Om':<10} {'Operational':<12} {'Checkpoint'}")
    print("-" * 100)
    
    for dataset_name, result in all_results.items():
        if 'error' in result:
            print(f"{dataset_name:<15} {'ERROR':<8} {result['error'][:30]:<30}")
        else:
            zero = '✅' if result['zero_omission'] else '❌'
            op = '✅' if result['operational'] else '❌'
            ckpt = str(FENCE_CHECKPOINT_DIR / dataset_name / FENCE_BACKBONE / "best_model.pt")
            print(f"{dataset_name:<15} {result['balls']:<8} {result['test_fnr']:<12.6f} {result['test_fpr']:<12.6f} {zero:<10} {op:<12} {ckpt}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = FENCE_LOG_DIR / f"test_all_datasets_{timestamp}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n  📊 Results saved to: {output_path}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()