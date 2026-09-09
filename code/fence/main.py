"""
FENCE Main Entry Point.

Runs training and evaluation on specified dataset with sample limit option.

Usage:
    # Test mode (5k samples)
    python code/fence/main.py --dataset fasdd --test_mode
    
    # Full training
    python code/fence/main.py --dataset fasdd
    
    # Specific dataset
    python code/fence/main.py --dataset dfire --test_mode
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from sklearn.model_selection import train_test_split
import argparse
import torch
import numpy as np
from tqdm import tqdm
import random
from datetime import datetime

# Import FENCE components
from fence.config import (
    DATASET_NAME, DEVICE, GPU_IDS, SEED,
    FENCE_BACKBONE, FENCE_EMBEDDING_DIM,
    FENCE_NUM_CLUSTERS, FENCE_MAX_CLUSTERS, FENCE_MIN_CLUSTER_SIZE,
    FENCE_RADIUS_MARGIN, FENCE_RADIUS_METHOD,
    FENCE_ABSTENTION_ENABLED,
    FENCE_BATCH_SIZE, FENCE_LEARNING_RATE, FENCE_WEIGHT_DECAY,
    FENCE_NUM_EPOCHS, FENCE_EVAL_EVERY_N_EPOCHS,
    FENCE_CHECKPOINT_DIR, FENCE_LOG_DIR,
    MSC_TEMPERATURE, MSC_LAMBDA,
    SPECTRAL_NORM_ENABLED
)

from fence.core.fence_model import FENCEModel
from fence.training.feasible_trainer import FeasibleTrainer
from fence.utils.metrics import print_metrics_table, compute_all_metrics

# Import dataset utilities
from train_datasets.dataset_factory import create_dataset, create_dataloaders
from train_datasets.fasdd_dataset import FASDDDataset
from train_datasets.dfire_dataset import DFireDataset
from train_datasets.ustc_smokers_dataset import USTCSmokeRSDataset
from train_datasets.flame_dataset import FLAMEDataset
# Optional: WandB
try:
    from code.pAUC_CRC.utils.wandb_logger import WandbLogger
except ImportError:
    WandbLogger = None


def parse_args():
    parser = argparse.ArgumentParser(description="FENCE Training")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME,
                        choices=["fasdd", "dfire", "ustc_smokers", "flame"],
                        help="Dataset to use")
    parser.add_argument("--test_mode", action="store_true",
                        help="Use only 5k samples for testing")
    parser.add_argument("--sample_limit", type=int, default=5000,
                        help="Number of samples to use in test mode")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (overrides config)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable WandB logging")
    return parser.parse_args()


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
    base_dir = Path(__file__).parent.parent.parent
    
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
                 test_mode: bool = False, sample_limit: int = 5000):
    """Load dataset and return samples with optional sample limit."""
    
    print(f"\n{'='*60}")
    print(f"LOADING DATASET: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Create dataset instance
    dataset = create_dataset(dataset_name, image_dir, annotation_dir)
    
    # Parse samples
    if dataset_name == "dfire":
        # D-Fire has separate train/test
        train_samples = dataset.parse_samples()
        test_samples = dataset.parse_test_samples()
        samples = train_samples + test_samples
        external_test = test_samples
    elif dataset_name == "ustc_smokers":
        # USTC_SmokeRS from HuggingFace
        samples = dataset.parse_samples()
        test_samples = dataset.parse_test_samples()
        samples = samples + test_samples
        external_test = test_samples
    else:
        samples = dataset.parse_samples()
        external_test = []
    
    print(f"  Total samples: {len(samples)}")
    pos = sum(1 for s in samples if s['label'] == 1)
    neg = len(samples) - pos
    print(f"  Positive: {pos}, Negative: {neg}")
    
    # Apply sample limit in test mode
    if test_mode and len(samples) > sample_limit:
        random.seed(SEED)
        # Try to keep class balance
        pos_samples = [s for s in samples if s['label'] == 1]
        neg_samples = [s for s in samples if s['label'] == 0]
        
        # Sample balanced subset
        pos_limit = min(len(pos_samples), sample_limit // 2)
        neg_limit = min(len(neg_samples), sample_limit - pos_limit)
        
        pos_subset = random.sample(pos_samples, pos_limit) if pos_limit > 0 else []
        neg_subset = random.sample(neg_samples, neg_limit) if neg_limit > 0 else []
        
        samples = pos_subset + neg_subset
        random.shuffle(samples)
        
        print(f"  ⚠️ TEST MODE: Limited to {len(samples)} samples")
        print(f"    Positive: {len(pos_subset)}, Negative: {len(neg_subset)}")
    
    # Split samples into train/val/cal/test
    # Use 70/15/15 split for train/val/cal (test is external or from split)
    from sklearn.model_selection import train_test_split
    
    # Separate positives and negatives for stratified split
    pos_samples = [s for s in samples if s['label'] == 1]
    neg_samples = [s for s in samples if s['label'] == 0]
    
    # Split positives: 70% train, 15% val, 15% cal (no test from internal split)
    pos_train, pos_temp = train_test_split(
        pos_samples, test_size=0.30, random_state=SEED
    )
    pos_val, pos_cal = train_test_split(
        pos_temp, test_size=0.50, random_state=SEED
    )
    
    # Split negatives: 70% train, 30% val (no cal, no test)
    neg_train, neg_val = train_test_split(
        neg_samples, test_size=0.30, random_state=SEED
    )
    
    # Combine
    train_samples = pos_train + neg_train
    val_samples = pos_val + neg_val
    cal_samples = pos_cal  # Positive-only calibration
    
    # Shuffle
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(cal_samples)
    
    splits = {
        'train': train_samples,
        'val': val_samples,
        'cal': cal_samples,
        'test': external_test if external_test else []  # External test if available
    }
    
    print(f"\n  Split sizes:")
    for split_name, split_samples in splits.items():
        pos_count = sum(1 for s in split_samples if s['label'] == 1)
        print(f"    {split_name}: {len(split_samples)} (pos: {pos_count})")
    
    # Create dataloaders
    dataloaders = create_dataloaders(splits, test_mode=False)
    
    # For datasets with external test set, use it
    if dataset_name in ["dfire", "ustc_smokers"]:
        from datasets.dataset_factory import create_test_dataloader
        test_loader = create_test_dataloader(dataset, test_mode=False)
    else:
        test_loader = dataloaders.get('test', None)
    
    return {
        'train_loader': dataloaders['train'],
        'val_loader': dataloaders['val'],
        'cal_loader': dataloaders.get('cal', None),
        'test_loader': test_loader,
        'splits': splits,
        'dataset': dataset
    }


def main():
    args = parse_args()
    
    # Set up
    set_seed(SEED)
    device = DEVICE
    print(f"\nDevice: {device}")
    print(f"GPUs: {GPU_IDS}")
    
    # Override epochs if specified
    epochs = args.epochs if args.epochs is not None else FENCE_NUM_EPOCHS
    
    # Get dataset paths
    image_dir, annotation_dir = get_dataset_paths(args.dataset)
    
    # Load dataset
    data = load_dataset(
        args.dataset, image_dir, annotation_dir,
        test_mode=args.test_mode,
        sample_limit=args.sample_limit
    )
    
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    
    # Get positive samples count for constraints
    pos_count = sum(1 for s in data['splits']['train'] if s['label'] == 1)
    print(f"\n  Training positives: {pos_count}")
    
    # Initialize FENCE model
    print(f"\n{'='*60}")
    print("INITIALIZING FENCE MODEL")
    print(f"{'='*60}")
    
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
    
    # Initialize union of balls with training embeddings (first pass)
    print("\n  Initializing union of balls from training data...")
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Extracting embeddings"):
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
    
    # Setup trainer
    print(f"\n{'='*60}")
    print("SETTING UP TRAINER")
    print(f"{'='*60}")
    
    # WandB
    wandb_logger = None
    if not args.no_wandb and WandbLogger is not None:
        wandb_logger = WandbLogger(run_name=f"fence_{args.dataset}")
    
    # Checkpoint directory
    checkpoint_dir = FENCE_CHECKPOINT_DIR / args.dataset / FENCE_BACKBONE
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = FeasibleTrainer(
        model=model,
        union_of_balls=model.union_of_balls,
        train_loader=train_loader,
        val_loader=val_loader,
        constraints=model.constraints,
        learning_rate=FENCE_LEARNING_RATE,
        weight_decay=FENCE_WEIGHT_DECAY,
        num_epochs=epochs,
        eval_freq=FENCE_EVAL_EVERY_N_EPOCHS,
        device=device,
        use_amp=False,
        save_dir=checkpoint_dir,
        wandb_logger=wandb_logger
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    
    # Train
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Test mode: {args.test_mode}")
    print(f"  Samples: {len(data['splits']['train'])}")
    print(f"  Epochs: {epochs}")
    print(f"  Balls: {len(model.union_of_balls)}")
    print("=" * 60 + "\n")
    
    history = trainer.train()
    
    # Final evaluation on test set
    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*60}")
    
    if test_loader is not None:
        # Get predictions on test set
        all_labels = []
        all_scores = []
        all_preds = []
        all_abstained = []
        all_contains = []
        all_in_shell = []
        
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(device)
                labels = labels.to(device)
                
                pred = model.predict(images)
                
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(pred['scores'].cpu().numpy())
                all_preds.extend(pred['prediction'].cpu().numpy())
                all_abstained.extend(pred['abstained'].cpu().numpy())
                all_contains.extend(pred['contains'].cpu().numpy())
                all_in_shell.extend(pred['in_shell'].cpu().numpy())
        
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
        
        # Print metrics
        print_metrics_table(metrics)
        
        # Save results
        results_file = checkpoint_dir / f"test_results_{args.dataset}.txt"
        with open(results_file, 'w') as f:
            f.write(f"FENCE Test Results - {args.dataset}\n")
            f.write("=" * 60 + "\n")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.6f}\n")
                elif isinstance(value, dict):
                    continue
                else:
                    f.write(f"{key}: {value}\n")
        print(f"\n  Results saved to: {results_file}")
        
        # Also save to WandB
        if wandb_logger:
            wandb_logger.log_test(metrics)
    
    # Finish
    if wandb_logger:
        wandb_logger.finish()
    
    print(f"\n{'='*60}")
    print("FENCE TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Dataset: {args.dataset}")
    print(f"  Best FNR: {trainer.best_val_fnr:.6f}")
    print(f"  Best FPR: {trainer.best_val_fpr:.6f}")
    print(f"  Balls: {len(model.union_of_balls)}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()