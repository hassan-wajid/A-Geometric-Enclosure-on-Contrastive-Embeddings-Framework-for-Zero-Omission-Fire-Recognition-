"""
Stage 1: Mean-Shifted Contrastive (MSC) Pretraining.

- Warmup + Cosine Annealing with Restarts (SimCLR-style)
- Saves checkpoints every epoch for resume
- Saves final pretrained.pt for Stage 2
- Resume functionality included

Usage:
    # Run Stage 1 fresh
    python code/fence/stage1_pretrain.py --dataset ustc_smokers --epochs 60 --batch_size 192
    
    # Resume from checkpoint
    python code/fence/stage1_pretrain.py --dataset ustc_smokers --resume checkpoints/fence/ustc_smokers/vit_large_patch14_224/stage1/resume_checkpoint.pt
"""

import os
import sys
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '1'
# Reduce CUDA fragmentation (helps avoid OOM when fine-tuning a large backbone).
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import numpy as np
import json

from fence.config import (
    DEVICE, SEED, FENCE_BACKBONE, FENCE_EMBEDDING_DIM,
    FENCE_BATCH_SIZE, FENCE_CHECKPOINT_DIR, DATASET_NAME,
    FENCE_WEIGHT_DECAY,
    MSC_TEMPERATURE, MSC_LAMBDA, STAGE1_COMPACTNESS_BETA,
    SPECTRAL_NORM_ENABLED, SPECTRAL_NORM_ITERATIONS, USE_AMP
)
from fence.core.fence_model import FENCEModel
from train_datasets.dataset_factory import create_dataset, create_dataloaders


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


def stage1_loss(embeddings, labels, temperature=0.1, compactness_beta=0.5):
    """
    Mean-Shifted Contrastive (MSC) Loss.
    
    Combines:
    1. Supervised Contrastive (SupCon) loss
    2. Compactness loss (pull positives toward mean)
    
    Args:
        embeddings: (B, D) feature embeddings (L2-normalized)
        labels: (B,) binary labels (1=positive, 0=negative)
        temperature: Temperature for contrastive loss (τ)
        compactness_beta: Weight for compactness term (β)
    
    Returns:
        total_loss, supcon_loss, compact_loss
    """
    # Compute the loss in fp32 even under AMP autocast for numerical stability.
    z = F.normalize(embeddings.float(), dim=1)
    
    # --- 1. SUPERVISED CONTRASTIVE ---
    sim = z @ z.T / temperature
    sim.fill_diagonal_(-1e9)
    labels_float = labels.float()
    same_label = (labels_float[:, None] == labels_float[None, :]).float()
    same_label.fill_diagonal_(0)
    logits = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp = logits.exp()
    denom = exp.sum(dim=1, keepdim=True)
    pos_count = same_label.sum(dim=1).clamp(min=1)
    log_prob = logits - denom.log()
    loss_supcon = -(same_label * log_prob).sum(dim=1) / pos_count
    loss_supcon = loss_supcon.mean()
    
    # --- 2. COMPACTNESS (Mean-Shifted) ---
    pos_mask = labels == 1
    if pos_mask.any():
        pos_embeddings = z[pos_mask]
        pos_mean = F.normalize(pos_embeddings.mean(dim=0, keepdim=True), dim=1)
        loss_compact = (1.0 - (pos_embeddings @ pos_mean.T)).mean()
    else:
        loss_compact = torch.tensor(0.0, device=embeddings.device)
    
    total_loss = loss_supcon + compactness_beta * loss_compact
    return total_loss, loss_supcon, loss_compact


class WarmupCosineWithRestarts:
    """
    Warmup + Cosine Decay with Periodic Restarts.
    
    SimCLR-style scheduler:
    - Warmup: linear increase for first N epochs
    - Cosine decay: smooth decay to min_lr
    - Restart: jump back to peak_lr at restart points
    """
    def __init__(self, optimizer, warmup_epochs=5, peak_lr=5e-4, 
                 min_lr=1e-6, restart_epochs=20, total_epochs=50):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.restart_epochs = restart_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
    def step(self, epoch):
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Warmup: linear increase from 0 to peak_lr
            progress = (epoch + 1) / self.warmup_epochs
            lr = progress * self.peak_lr
            
        else:
            # Cosine decay with restarts
            cycle_epoch = epoch - self.warmup_epochs
            cycle = cycle_epoch // self.restart_epochs
            cycle_progress = (cycle_epoch % self.restart_epochs) / self.restart_epochs
            
            # Peak LR decays slightly each cycle
            decay_factor = 0.8 ** cycle
            current_peak = self.peak_lr * decay_factor
            
            # Cosine decay
            cos_value = np.cos(cycle_progress * np.pi)
            lr = self.min_lr + (current_peak - self.min_lr) * (1 + cos_value) / 2
            
        lr = max(self.min_lr, min(self.peak_lr, lr))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_scale', 1.0)

        return lr


def load_stage1_checkpoint(checkpoint_path: Path, model, optimizer, device):
    """Load checkpoint and return (model, optimizer, start_epoch, history)."""
    # weights_only=False: our own checkpoint stores numpy scalars in `history`,
    # which PyTorch 2.6's default (weights_only=True) refuses to unpickle.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load training state
    start_epoch = checkpoint['epoch'] + 1
    history = checkpoint.get('history', {
        'loss': [], 'supcon_loss': [], 'compact_loss': [], 'lr': []
    })
    
    print(f"  📥 Resumed from: {checkpoint_path}")
    print(f"  Resuming at epoch: {start_epoch}")
    print(f"  Best loss so far: {min(history['loss']) if history['loss'] else 'N/A'}")
    
    return model, optimizer, start_epoch, history


def stage1_pretrain(dataset_name: str, 
                    epochs: int = 50,
                    learning_rate: float = 5e-4,  # <-- KEEP THIS PARAMETER NAME
                    warmup_epochs: int = 5,
                    min_lr: float = 1e-6,
                    restart_epochs: int = 20,
                    sample_limit: int = None,
                    batch_size: int = None,
                    resume_from: str = None):
    """
    Stage 1: MSC Pretraining with Warmup + Cosine Restarts.
    """
    # Optional batch-size override (the dataloader reads root config.BATCH_SIZE).
    # NOTE: each step processes 2x this many images (two contrastive views).
    if batch_size is not None:
        import config as _root_cfg
        _root_cfg.BATCH_SIZE = batch_size
        print(f"  Overriding batch size -> {batch_size} (effective {2*batch_size} with two views)")
    print("\n" + "="*60)
    print(f"STAGE 1: MSC PRETRAINING")
    print(f"  Dataset: {dataset_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Peak LR: {learning_rate}")
    print(f"  Min LR: {min_lr}")
    print(f"  Warmup: {warmup_epochs} epochs")
    print(f"  Restarts: every {restart_epochs} epochs")
    print("="*60 + "\n")
    
    set_seed(SEED)
    device = DEVICE
    
    # Load dataset
    image_dir, annotation_dir = get_dataset_paths(dataset_name)
    dataset = create_dataset(dataset_name, image_dir, annotation_dir)
    samples = dataset.parse_samples()
    
    if sample_limit and len(samples) > sample_limit:
        import random
        random.seed(SEED)
        pos_samples = [s for s in samples if s['label'] == 1]
        neg_samples = [s for s in samples if s['label'] == 0]
        pos_limit = min(len(pos_samples), sample_limit // 2)
        neg_limit = min(len(neg_samples), sample_limit - pos_limit)
        samples = random.sample(pos_samples, pos_limit) + random.sample(neg_samples, neg_limit)
        random.shuffle(samples)
        print(f"  Limited to {len(samples)} samples")
    
    splits = dataset.split_samples(samples)
    dataloaders = create_dataloaders({'train': splits['train']}, test_mode=False)
    train_loader = dataloaders['train']
    
    print(f"  Train samples: {len(splits['train'])}")
    print(f"  Train batches: {len(train_loader)}")
    
    # Initialize model
    model = FENCEModel(
        backbone_name=FENCE_BACKBONE,
        embedding_dim=FENCE_EMBEDDING_DIM,
        num_balls=0,
        margin=0.1,
        temperature=MSC_TEMPERATURE,
        lambda_class=MSC_LAMBDA,
        use_mean_shift=True,
        spectral_norm=SPECTRAL_NORM_ENABLED,
        device=device
    )
    
    # Freeze classifier head (only backbone + projection needed for Stage 1)
    # for param in model.backbone.classifier.parameters():
    #     param.requires_grad = False
    
    # Separate parameter groups: fine-tune the pretrained CLIP backbone with a
    # smaller LR than the freshly-initialized projection head / classifier.
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'clip_model' in name:          # CLIP vision tower (unfrozen blocks)
            backbone_params.append(p)
        else:                             # projection head + classifier
            head_params.append(p)

    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr_scale': 0.1})
    if head_params:
        param_groups.append({'params': head_params, 'lr_scale': 1.0})

    optimizer = optim.AdamW(
        param_groups,
        lr=min_lr,  # placeholder; scheduler sets the real LR at the top of each epoch
        weight_decay=FENCE_WEIGHT_DECAY
    )

    n_backbone = sum(p.numel() for p in backbone_params)
    n_head = sum(p.numel() for p in head_params)
    print(f"  Trainable backbone params: {n_backbone:,} (lr_scale=0.1)")
    print(f"  Trainable head params:     {n_head:,} (lr_scale=1.0)")
    if n_backbone == 0:
        print("  ⚠️  WARNING: 0 trainable backbone params — backbone will NOT learn!")
    # Checkpoint directory
    checkpoint_dir = FENCE_CHECKPOINT_DIR / dataset_name / FENCE_BACKBONE / "stage1"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume if requested
    start_epoch = 0
    history = {'loss': [], 'supcon_loss': [], 'compact_loss': [], 'lr': []}
    
    if resume_from and Path(resume_from).exists():
        model, optimizer, start_epoch, history = load_stage1_checkpoint(
            Path(resume_from), model, optimizer, device
        )
    
    # Learning rate scheduler
    lr_scheduler = WarmupCosineWithRestarts(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        peak_lr=learning_rate,  # <-- Use learning_rate as peak_lr
        min_lr=min_lr,
        restart_epochs=restart_epochs,
        total_epochs=epochs
    )
    
    # Mixed-precision scaler (halves activation memory + speeds up the backbone).
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # Training loop
    print(f"\n  Starting training from epoch {start_epoch+1}...")
    print(f"  Mixed precision (AMP): {USE_AMP}")

    for epoch in range(start_epoch, epochs):
        # Set the LR for THIS epoch BEFORE training (warmup -> cosine -> restarts).
        # (Previously the scheduler stepped AFTER the epoch, so epoch 1 ran at
        #  min_lr=1e-6 and every epoch used the previous epoch's LR.)
        current_lr = lr_scheduler.step(epoch)

        model.train()
        total_loss = 0.0
        total_supcon = 0.0
        total_compact = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_data in pbar:
            # --- HANDLE TWO VIEWS (FIX 3) ---
            if len(batch_data) == 4:
                images1, images2, labels, global_indices = batch_data
            else:
                images1, labels = batch_data
                images2 = images1  # Fallback
                global_indices = torch.arange(len(labels), device=device)
            
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels = labels.to(device)
            global_indices = global_indices.to(device)
            
            # --- CONCATENATE TWO VIEWS (FIX 3) ---
            images = torch.cat([images1, images2], dim=0)
            labels_concat = torch.cat([labels, labels], dim=0)
            
            # Forward pass (autocast for the heavy backbone; loss stays fp32).
            # Pass labels=None: we compute stage1_loss externally, so we don't
            # want the model's internal (unused) MSC loss graph eating memory.
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(images)
                embeddings = outputs['embeddings']

            # --- MSC LOSS ---
            loss, supcon_loss, compact_loss = stage1_loss(
                embeddings, labels_concat,
                temperature=0.1,
                compactness_beta=0.5
            )

            # Backward (scaled for AMP)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_supcon += supcon_loss.item()
            total_compact += compact_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'supcon': f'{supcon_loss.item():.4f}',
                'compact': f'{compact_loss.item():.4f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_supcon = total_supcon / num_batches
        avg_compact = total_compact / num_batches
        
        history['loss'].append(avg_loss)
        history['supcon_loss'].append(avg_supcon)
        history['compact_loss'].append(avg_compact)
        history['lr'].append(current_lr)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Loss: {avg_loss:.4f} (SupCon: {avg_supcon:.4f}, Compact: {avg_compact:.4f})")
        print(f"  LR: {current_lr:.8f}")
        
        # Check if LR restarted
        if len(history['lr']) > 1 and current_lr > history['lr'][-2] * 1.5:
            print(f"  🔄 LR Restart! (jumped to {current_lr:.8f})")
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'history': history,
            'config': {
                'backbone': FENCE_BACKBONE,
                'embedding_dim': FENCE_EMBEDDING_DIM,
                'temperature': 0.1,
                'compactness_beta': 0.5,
                'peak_lr': learning_rate,
                'min_lr': min_lr,
                'warmup_epochs': warmup_epochs,
                'restart_epochs': restart_epochs
            }
        }
        
        resume_path = checkpoint_dir / "resume_checkpoint.pt"
        torch.save(checkpoint, resume_path)
    
    # --- SAVE FINAL PRETRAINED MODEL ---
    checkpoint_path = checkpoint_dir / "pretrained.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'epoch': epochs,
        'config': {
            'backbone': FENCE_BACKBONE,
            'embedding_dim': FENCE_EMBEDDING_DIM,
            'temperature': MSC_TEMPERATURE,
            'compactness_beta': STAGE1_COMPACTNESS_BETA,
            'peak_lr': learning_rate,
            'min_lr': min_lr,
            'warmup_epochs': warmup_epochs,
            'restart_epochs': restart_epochs
        }
    }, checkpoint_path)
    
    # Save history as JSON
    history_path = checkpoint_dir / "history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Stage 1 complete!")
    print(f"   Final SupCon: {avg_supcon:.4f}, Compact: {avg_compact:.4f}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   History: {history_path}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Stage 1 MSC Pretraining")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME,
                        choices=["fasdd", "dfire", "ustc_smokers", "flame"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--restart_epochs", type=int, default=20)
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-view batch size (effective batch = 2x this). "
                             "Lower this if you hit CUDA OOM.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    args = parser.parse_args()

    stage1_pretrain(
        dataset_name=args.dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        restart_epochs=args.restart_epochs,
        sample_limit=args.sample_limit,
        batch_size=args.batch_size,
        resume_from=args.resume
    )


if __name__ == "__main__":
    main()