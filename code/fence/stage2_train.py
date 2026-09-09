"""
Stage 2: Feasible Learning Training.

- Loads Stage 1 pretrained backbone
- Persistent per-sample multipliers
- Two-radius system (quantile training, max deployment)
- Resume functionality
- Finalization with quarantine and verification

Usage:
    # Run Stage 2 fresh (with Stage 1 checkpoint)
    python code/fence/stage2_train.py --dataset ustc_smokers --batch_size 32 --epochs 1
    
    # Resume from Stage 2 checkpoint
    python code/fence/stage2_train.py --dataset ustc_smokers --resume checkpoints/fence/ustc_smokers/vit_large_patch14_224/stage2/epoch_005.pt --epochs 30
"""

import os
import sys
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import numpy as np
import time
import json

# Import from root config
from config import DEVICE, SEED
from fence.config import (
    FENCE_BACKBONE, FENCE_EMBEDDING_DIM,
    FENCE_NUM_CLUSTERS, FENCE_MAX_CLUSTERS, FENCE_MIN_CLUSTER_SIZE,
    FENCE_RADIUS_MARGIN, FENCE_RADIUS_METHOD, FENCE_RADIUS_QUANTILE,
    FENCE_LEARNING_RATE, FENCE_WEIGHT_DECAY, FENCE_NUM_EPOCHS,
    FENCE_EVAL_EVERY_N_EPOCHS, FENCE_CHECKPOINT_DIR,
    MSC_TEMPERATURE, MSC_LAMBDA, SPECTRAL_NORM_ENABLED,
    FPR_TEMPERATURE, DATASET_NAME, USE_AMP
)
from fence.core.fence_model import FENCEModel
from fence.training.feasible_trainer import FeasibleTrainer, unpack_batch
from fence.utils.metrics import print_metrics_table
from train_datasets.dataset_factory import create_dataset, create_dataloaders, auto_splits


def set_seed(seed: int = SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataset_paths(dataset_name: str):
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


def load_stage1_checkpoint(model, checkpoint_path: Path):
    """Load Stage 1 pretrained weights."""
    if not checkpoint_path.exists():
        print(f"  ⚠️ Stage 1 checkpoint not found: {checkpoint_path}")
        print("  Starting from scratch.")
        return model
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Loaded Stage 1 weights from: {checkpoint_path}")
    print(f"  ✓ Epochs trained: {checkpoint.get('epoch', 'N/A')}")
    return model


def restore_union_from_dict(union, union_dict):
    """Restore UnionOfBalls from dict."""
    from fence.geometry.ball import Ball
    
    if not union_dict:
        return union
    
    union.balls = []
    for ball_data in union_dict.get('balls', []):
        center = torch.tensor(ball_data['center'], device=union.device)
        ball = Ball(center, radius=ball_data['radius'], margin=ball_data['margin'])
        ball._assigned_indices = ball_data.get('assigned_indices', [])
        union.balls.append(ball)
    
    union._feature_dim = union_dict.get('feature_dim', None)
    if 'centers_ema' in union_dict:
        union._centers_ema = torch.tensor(union_dict['centers_ema'], device=union.device)
    
    print(f"  ✓ Restored {len(union.balls)} balls from checkpoint")
    return union


def stage2_train(dataset_name: str,
                 stage1_checkpoint: Path = None,
                 epochs: int = 30,
                 learning_rate: float = 1e-5,
                 sample_limit: int = None,
                 batch_size: int = None,
                 use_quantile_radius: bool = True,
                 freeze_backbone: bool = True,
                 finalize_split: str = 'train',
                 num_clusters: int = None,
                 auto_resume: bool = False,
                 resume_from: str = None):
    """
    Stage 2: Feasible Learning with resume.
    """
    # Optional batch-size override (dataloader reads root config.BATCH_SIZE).
    if batch_size is not None:
        import config as _root_cfg
        _root_cfg.BATCH_SIZE = batch_size
        print(f"  Overriding batch size -> {batch_size}")

    set_seed(SEED)
    device = DEVICE

    # --- Checkpoint dirs (computed up-front so paths are automatic) ---
    checkpoint_dir = FENCE_CHECKPOINT_DIR / dataset_name / FENCE_BACKBONE / "stage2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Auto-find the Stage 1 checkpoint for this dataset (no manual path needed).
    if stage1_checkpoint is None:
        default_s1 = FENCE_CHECKPOINT_DIR / dataset_name / FENCE_BACKBONE / "stage1" / "pretrained.pt"
        if default_s1.exists():
            stage1_checkpoint = default_s1
            print(f"  Auto-found Stage 1 checkpoint: {stage1_checkpoint}")
        else:
            print(f"  ⚠️ No Stage 1 checkpoint at default location: {default_s1}")

    # Auto-resume: pick the latest Stage 2 checkpoint if asked and none was given.
    if auto_resume and resume_from is None:
        cands = sorted(checkpoint_dir.glob("epoch_*.pt"))
        if cands:
            resume_from = str(cands[-1])
        elif (checkpoint_dir / "best_model.pt").exists():
            resume_from = str(checkpoint_dir / "best_model.pt")
        if resume_from:
            print(f"  Auto-resuming from: {resume_from}")

    print("\n" + "="*60)
    print(f"STAGE 2: FEASIBLE LEARNING")
    print(f"  Dataset: {dataset_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Quantile Radius: {use_quantile_radius}")
    print(f"  Freeze backbone: {freeze_backbone}")
    if resume_from:
        print(f"  Resuming from: {resume_from}")
    print("="*60 + "\n")

    # Load dataset + AUTO split (decides val/test per dataset automatically)
    image_dir, annotation_dir = get_dataset_paths(dataset_name)
    dataset = create_dataset(dataset_name, image_dir, annotation_dir)
    splits = auto_splits(dataset, sample_limit=sample_limit)

    dataloaders = create_dataloaders(
        {'train': splits['train'], 'val': splits['val']},
        test_mode=False
    )
    train_loader = dataloaders['train']      # augmented + shuffled (gradient steps)
    val_loader = dataloaders['val']

    # Deterministic loader over the SAME train split (val transforms, no shuffle).
    # ALL geometry (union fit, recompute, finalize, verify) must use CLEAN, non-
    # augmented embeddings so the zero-FNR certificate matches deployment inputs.
    geom_loader = create_dataloaders({'geom': splits['train']}, test_mode=False)['geom']

    print(f"  Train samples: {len(splits['train'])}")
    print(f"  Val samples: {len(splits['val'])}")
    print(f"  Train batches: {len(train_loader)}")
    
    # Number of balls used to tile the positive cluster. 0 = auto-detect.
    # Use several (e.g. 3-10) moderate balls instead of one big one; far outliers
    # still get personal balls. More balls help multi-modal positive manifolds.
    n_balls = num_clusters if num_clusters is not None else FENCE_NUM_CLUSTERS

    # Initialize model
    model = FENCEModel(
        backbone_name=FENCE_BACKBONE,
        embedding_dim=FENCE_EMBEDDING_DIM,
        num_balls=n_balls,
        max_balls=FENCE_MAX_CLUSTERS,
        min_cluster_size=FENCE_MIN_CLUSTER_SIZE,
        margin=FENCE_RADIUS_MARGIN,
        temperature=MSC_TEMPERATURE,
        lambda_class=MSC_LAMBDA,
        use_mean_shift=True,
        spectral_norm=SPECTRAL_NORM_ENABLED,
        device=device
    )
    
    # --- HANDLE RESUME FIRST ---
    start_epoch = 0
    resume_state = None
    skip_union_init = False
    
    if resume_from and Path(resume_from).exists():
        print(f"\n  📥 Loading resume checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        
        # 1. Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  ✓ Model loaded")
        
        # 2. Restore union of balls
        if 'union_of_balls' in checkpoint:
            restore_union_from_dict(model.union_of_balls, checkpoint['union_of_balls'])
        
        # 3. Store training state for later
        resume_state = {
            'optimizer': checkpoint['optimizer_state_dict'],
            'scheduler': checkpoint.get('scheduler_state_dict', None),
            'epoch': checkpoint['epoch'],
            'best_val_fnr': checkpoint.get('best_val_fnr', 1.0),
            'best_val_fpr': checkpoint.get('best_val_fpr', 1.0),
            'best_epoch': checkpoint.get('best_epoch', -1),
            'history': checkpoint.get('history', {}),
            'multipliers': checkpoint.get('multipliers', None)
        }
        start_epoch = resume_state['epoch'] + 1
        
        print(f"  Resuming at epoch: {start_epoch}")
        print(f"  Best FNR so far: {resume_state['best_val_fnr']:.6f}")
        print(f"  Best FPR so far: {resume_state['best_val_fpr']:.6f}")
        
        skip_union_init = True
    else:
        print("  Fresh start (no resume checkpoint found)")
    
    # --- Load Stage 1 weights (only if not resuming) ---
    if not skip_union_init:
        if stage1_checkpoint:
            load_stage1_checkpoint(model, Path(stage1_checkpoint))
        else:
            print("  ⚠️ No Stage 1 checkpoint provided. Starting from scratch.")

    # --- FREEZE the embedding backbone ---
    # Stage 1 already produced an optimal, well-separated representation. Stage 2's
    # geometric objective has a DEGENERATE optimum (collapse everything together),
    # and fine-tuning the backbone slides into it (we saw FPR jump to 1.0 and 90%
    # abstention). With the embeddings frozen, the geometry is fit on a FIXED, good
    # representation and cannot collapse. Only the union of balls + multipliers
    # (and the tiny auxiliary classifier) adapt.
    if freeze_backbone:
        for p in model.backbone.backbone.parameters():
            p.requires_grad = False
        n_frozen = sum(p.numel() for p in model.backbone.backbone.parameters())
        print(f"  🔒 Froze embedding backbone ({n_frozen:,} params) — Stage 1 representation preserved.")

    # --- Extract CLEAN embeddings (+ global indices) from a loader ---
    def extract_clean(desc, loader=None):
        loader = loader if loader is not None else geom_loader
        embs, labs, idxs = [], [], []
        model.eval()
        with torch.no_grad():
            for batch_data in tqdm(loader, desc=desc):
                images, labels, gidx = unpack_batch(batch_data, device)
                images = images.to(device)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    outputs = model(images)
                embs.append(outputs['embeddings'].float())
                labs.append(labels.to(device))
                idxs.append(gidx.to(device))
        return torch.cat(embs), torch.cat(labs), torch.cat(idxs)

    all_embeddings, all_labels_full, all_global_idx = extract_clean("  Extracting (clean)")

    # --- Initialize union if not resuming ---
    if not skip_union_init:
        method = 'quantile' if use_quantile_radius else 'max'
        print(f"\n  Initializing union with {method} radius...")
        model.fit_union(all_embeddings, all_labels_full, radius_method=method)
        print(f"  Balls initialized: {len(model.union_of_balls)}")

    # Store OOD reference (clean embeddings) in both fresh and resume paths
    model.store_ood_reference(all_embeddings, all_labels_full)

    # Get GLOBAL positive indices for the persistent multipliers
    pos_mask_full = all_labels_full == 1
    pos_indices = all_global_idx[pos_mask_full]
    n_pos = len(pos_indices)

    # Initialize constraints and multipliers
    model.init_constraints(n_pos)

    print(f"\n  Checkpoint directory: {checkpoint_dir}")

    # Trainer
    trainer = FeasibleTrainer(
        model=model,
        union_of_balls=model.union_of_balls,
        train_loader=train_loader,
        val_loader=val_loader,
        eval_loader=geom_loader,
        learning_rate=learning_rate,
        weight_decay=FENCE_WEIGHT_DECAY,
        num_epochs=epochs,
        eval_freq=FENCE_EVAL_EVERY_N_EPOCHS,
        device=device,
        use_amp=USE_AMP,
        save_dir=checkpoint_dir,
        wandb_logger=None,
        margin=FENCE_RADIUS_MARGIN,
        fpr_weight=0.5,
        fpr_temperature=FPR_TEMPERATURE
    )
    
    # Initialize multipliers with positive indices
    if n_pos > 0:
        trainer.initialize_multipliers(pos_indices.cpu(), n_pos)
    
    # --- Apply resume state to trainer ---
    if resume_state:
        # Load optimizer state
        trainer.optimizer.load_state_dict(resume_state['optimizer'])
        # Load scheduler state
        if resume_state['scheduler'] and trainer.scheduler:
            trainer.scheduler.load_state_dict(resume_state['scheduler'])
        # Load training state
        trainer.current_epoch = start_epoch
        trainer.best_val_fnr = resume_state['best_val_fnr']
        trainer.best_val_fpr = resume_state['best_val_fpr']
        trainer.best_epoch = resume_state['best_epoch']
        trainer.history = resume_state['history']
        # Load multipliers state
        if resume_state['multipliers'] and trainer.multipliers:
            m = resume_state['multipliers']
            trainer.multipliers.lambdas = m['lambdas'].to(trainer.device)
            trainer.multipliers.integrals = m['integrals'].to(trainer.device)
            trainer.multipliers.prev_g = m['prev_g'].to(trainer.device)
            print("  ✓ Restored multipliers state")
    
    # Train
    print(f"\n  Starting training from epoch {trainer.current_epoch+1 if trainer.current_epoch > 0 else 1}...")
    start_time = time.time()
    history = trainer.train()
    train_time = time.time() - start_time
    
    # --- FINALIZE AND VERIFY (on CLEAN embeddings) ---
    # The zero-FNR certificate holds over the FINALIZE set (every positive there
    # gets enclosed — far ones get their own personal ball). Choose what that set
    # is: 'train' (held-out val stays a true generalization test) or 'all' (certify
    # FNR=0 over every labeled positive you have, incl. val/cal — transductive).
    if finalize_split == 'all':
        final_samples = (splits['train'] + splits.get('val', []) + splits.get('test', []))
        final_loader = create_dataloaders({'final': final_samples}, test_mode=False)['final']
        print(f"\n  Finalizing on ALL labeled data ({len(final_samples)} samples)...")
    else:
        final_loader = geom_loader
        print(f"\n  Finalizing on TRAIN split...")

    all_embeddings, all_labels_full, all_global_idx = extract_clean(
        "  Finalizing (clean)", loader=final_loader)

    # Finalize with deployment radii (max + margin) + a personal ball for every
    # otherwise-uncovered positive. Pass global indices so quarantine slack aligns.
    model.finalize(all_embeddings, all_labels_full, trainer.multipliers,
                   global_indices=all_global_idx)

    # Verify the zero-FNR certificate on the finalize set
    print(f"\n  Verifying zero FNR on the finalize set...")
    verified = model.verify_zero_fnr(all_embeddings, all_labels_full)
    if not verified:
        print("  ⚠️ WARNING: Zero FNR certificate FAILED!")
    else:
        print("  ✅ Zero FNR certificate PASSED!")

    # Save final model — INCLUDING quarantine balls (personal balls for far
    # positives), otherwise the coverage guarantee is lost at eval time.
    final_path = checkpoint_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'union_of_balls': model.union_of_balls.to_dict(),
        'quarantine_balls': [b.to_dict() for b in model.quarantine_balls],
        'margin': float(model.margin),
        'verified': verified,
        'history': history,
        'best_val_fnr': trainer.best_val_fnr,
        'best_val_fpr': trainer.best_val_fpr,
        'epoch': epochs
    }, final_path)
    print(f"\n  💾 Final model saved to: {final_path}")
    
    # Save summary
    summary_path = checkpoint_dir / "stage2_summary.json"
    summary = {
        'dataset': dataset_name,
        'epochs': epochs,
        'train_time': train_time,
        'samples': len(splits['train']),
        'val_samples': len(splits['val']),
        'balls': len(model.union_of_balls),
        'best_val_fnr': trainer.best_val_fnr,
        'best_val_fpr': trainer.best_val_fpr,
        'verified': verified,
        'quarantined': len(getattr(model, '_quarantine_indices', [])),
        'checkpoint': str(final_path)
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  📊 Summary saved to: {summary_path}")
    
    return model, history, verified


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Feasible Learning")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME,
                        choices=["fasdd", "dfire", "ustc_smokers", "flame"],
                        help="Dataset to use")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Path to Stage 1 pretrained weights")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--sample_limit", type=int, default=None,
                        help="Limit samples (for testing)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (lower if you hit CUDA OOM)")
    parser.add_argument("--no_quantile", action="store_true",
                        help="Use max radius instead of quantile during training")
    parser.add_argument("--unfreeze_backbone", action="store_true",
                        help="Fine-tune the backbone in Stage 2 (NOT recommended — "
                             "Stage 1 embeddings are already optimal and fine-tuning "
                             "can collapse them).")
    parser.add_argument("--num_clusters", type=int, default=None,
                        help="Number of positive balls (0=auto-detect). Try 3-10 to "
                             "tile the cluster with several moderate balls instead of "
                             "one big one. Far outliers still get personal balls.")
    parser.add_argument("--finalize_split", type=str, default="train",
                        choices=["train", "all"],
                        help="Set the zero-FNR certificate covers. 'train' keeps val "
                             "as a held-out test; 'all' encloses every labeled positive "
                             "(train+val+cal) -> FNR=0 everywhere (transductive).")
    parser.add_argument("--auto_resume", action="store_true",
                        help="Resume from the latest Stage 2 checkpoint automatically")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a specific Stage 2 checkpoint path")
    args = parser.parse_args()

    stage2_train(
        dataset_name=args.dataset,
        stage1_checkpoint=args.stage1_checkpoint,   # None => auto-found
        epochs=args.epochs,
        learning_rate=args.lr,
        sample_limit=args.sample_limit,
        batch_size=args.batch_size,
        use_quantile_radius=not args.no_quantile,
        freeze_backbone=not args.unfreeze_backbone,
        finalize_split=args.finalize_split,
        num_clusters=args.num_clusters,
        auto_resume=args.auto_resume,
        resume_from=args.resume
    )


if __name__ == "__main__":
    main()