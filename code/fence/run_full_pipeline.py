"""
Full FENCE Pipeline - Stage 1 + Stage 2.

Runs:
1. Stage 1: MSC pretraining (contrastive + compactness)
2. Stage 2: Feasible Learning (persistent multipliers, two-radius system)
3. Final evaluation

Usage:
    python code/fence/run_full_pipeline.py --dataset ustc_smokers --stage1_epochs 50 --stage2_epochs 30
    
    # Quick test mode (5k samples, fewer epochs)
    python code/fence/run_full_pipeline.py --dataset ustc_smokers --test_mode
"""

import os
import sys
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import argparse
import torch
import json
from datetime import datetime

from fence.config import (
    DEVICE, SEED, FENCE_BACKBONE, FENCE_EMBEDDING_DIM,
    FENCE_CHECKPOINT_DIR, DATASET_NAME,
    FENCE_NUM_EPOCHS, FENCE_LEARNING_RATE
)
from fence.stage1_pretrain import stage1_pretrain
from fence.stage2_train import stage2_train
from fence.utils.metrics import print_metrics_table


def set_seed(seed: int = SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Full FENCE Pipeline")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME,
                        choices=["fasdd", "dfire", "ustc_smokers", "flame"],
                        help="Dataset to use")
    parser.add_argument("--test_mode", action="store_true",
                        help="Use 5k samples and fewer epochs")
    parser.add_argument("--stage1_epochs", type=int, default=50,
                        help="Stage 1 epochs")
    parser.add_argument("--stage2_epochs", type=int, default=30,
                        help="Stage 2 epochs")
    parser.add_argument("--stage1_lr", type=float, default=5e-4,
                        help="Stage 1 learning rate")
    parser.add_argument("--stage2_lr", type=float, default=1e-5,
                        help="Stage 2 learning rate")
    parser.add_argument("--skip_stage1", action="store_true",
                        help="Skip Stage 1 (use random init)")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Path to existing Stage 1 checkpoint")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("FULL FENCE PIPELINE")
    print(f"  Dataset: {args.dataset}")
    print(f"  Test Mode: {args.test_mode}")
    print(f"  Stage 1 Epochs: {args.stage1_epochs}")
    print(f"  Stage 2 Epochs: {args.stage2_epochs}")
    print("="*60 + "\n")
    
    set_seed(SEED)
    
    sample_limit = 5000 if args.test_mode else None
    stage1_epochs = 10 if args.test_mode else args.stage1_epochs
    stage2_epochs = 10 if args.test_mode else args.stage2_epochs
    
    checkpoint_dir = FENCE_CHECKPOINT_DIR / args.dataset / FENCE_BACKBONE
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stage1_checkpoint_path = checkpoint_dir / "stage1" / "pretrained.pt"
    
    # ----- STAGE 1: MSC Pretraining -----
    stage1_model = None
    if not args.skip_stage1:
        if args.stage1_checkpoint and Path(args.stage1_checkpoint).exists():
            print(f"  Using existing Stage 1 checkpoint: {args.stage1_checkpoint}")
            stage1_checkpoint_path = Path(args.stage1_checkpoint)
        else:
            print("\n" + "#"*60)
            print("# STAGE 1: MSC Pretraining")
            print("#"*60)
            
            stage1_model, stage1_history = stage1_pretrain(
                dataset_name=args.dataset,
                epochs=stage1_epochs,
                learning_rate=args.stage1_lr,
                sample_limit=sample_limit
            )
            
            # The model is saved inside stage1_pretrain, but we keep it in memory too
            stage1_checkpoint_path = checkpoint_dir / "stage1" / "pretrained.pt"
            print(f"\n  ✅ Stage 1 complete. Checkpoint: {stage1_checkpoint_path}")
    else:
        print("\n  ⚠️ Skipping Stage 1 (using random init)")
        stage1_checkpoint_path = None
    
    # ----- STAGE 2: Feasible Learning -----
    print("\n" + "#"*60)
    print("# STAGE 2: Feasible Learning")
    print("#"*60)
    
    # If we have the stage1_model in memory, we could pass it directly.
    # But stage2_train loads from checkpoint, so we pass the path.
    # For efficiency, we could modify stage2_train to accept a model directly.
    # For now, passing the checkpoint path is fine.
    
    model, history, verified = stage2_train(
        dataset_name=args.dataset,
        stage1_checkpoint=stage1_checkpoint_path,
        epochs=stage2_epochs,
        learning_rate=args.stage2_lr,
        sample_limit=sample_limit,
        use_quantile_radius=True
    )
    
    # ----- SUMMARY -----
    print("\n" + "="*60)
    print("FULL PIPELINE COMPLETE")
    print("="*60)
    print(f"  Dataset: {args.dataset}")
    print(f"  Verified: {verified}")
    print(f"  Checkpoint: {FENCE_CHECKPOINT_DIR / args.dataset / FENCE_BACKBONE / 'stage2' / 'final_model.pt'}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()