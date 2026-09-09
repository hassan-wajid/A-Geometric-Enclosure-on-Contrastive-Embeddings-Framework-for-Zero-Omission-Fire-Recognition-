"""
Feasible Learning Trainer with Persistent Multipliers and Two-Radius System.

Implements the full FENCE training loop with:
- Persistent per-sample Lagrange multipliers (νPI controller)
- Two-radius system (quantile for training, max for deployment)
- Differentiable FPR loss (hinge-based)
- Alternating optimization: backbone gradients + multiplier updates
"""

import os
import sys
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import time

try:
    from fence.training.multipliers import PersistentMultipliers
    from fence.utils.metrics import compute_all_metrics, print_metrics_table
except ImportError:
    from training.multipliers import PersistentMultipliers
    from utils.metrics import compute_all_metrics, print_metrics_table


def unpack_batch(batch_data, device):
    """
    Unpack a dataloader batch into (images, labels, global_indices).

    The dataset yields a 4-tuple (view1, view2, label, global_idx). Stage 2 only
    needs ONE view + the GLOBAL index (for the persistent per-sample multipliers).
    Falls back gracefully for older 3-/2-tuple formats.
    """
    if len(batch_data) == 4:
        images, _view2, labels, global_indices = batch_data
    elif len(batch_data) == 3:
        images, labels, global_indices = batch_data
    else:
        images, labels = batch_data
        global_indices = torch.arange(len(labels), device=device)
    return images, labels, global_indices


class FeasibleTrainer:
    """
    Feasible Learning Trainer with persistent multipliers.
    """
    
    def __init__(self,
                 model,
                 union_of_balls,
                 train_loader,
                 val_loader,
                 eval_loader=None,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 num_epochs: int = 15,
                 eval_freq: int = 1,
                 device: str = 'cpu',
                 use_amp: bool = False,
                 save_dir: Optional[Path] = None,
                 wandb_logger = None,
                 margin: float = 0.1,
                 fpr_weight: float = 0.5,
                 fpr_temperature: float = 10.0):

        self.model = model
        self.union_of_balls = union_of_balls
        self.train_loader = train_loader
        self.val_loader = val_loader
        # Deterministic (val-transform, unshuffled) loader over the TRAIN split,
        # used for fitting the geometry on CLEAN embeddings. Falls back to the
        # augmented train_loader if not provided.
        self.eval_loader = eval_loader if eval_loader is not None else train_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.eval_freq = eval_freq
        self.device = device
        self.use_amp = use_amp
        self.save_dir = save_dir
        self.wandb = wandb_logger
        self.margin = margin
        self.fpr_weight = fpr_weight
        self.fpr_temperature = fpr_temperature
        
        # Multipliers (initialized later with positive indices)
        self.multipliers = None
        self.pos_indices_map = None
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=learning_rate * 0.01
        )

        # Mixed-precision scaler (no-op when use_amp=False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # Training state
        self.current_epoch = 0
        self.best_val_fnr = 1.0
        self.best_val_fpr = 1.0
        self.best_score = float('inf')   # best (FNR+FPR) seen — for model selection
        self.best_epoch = -1
        
        # History
        self.history = {
            'train_loss': [],
            'train_fpr_loss': [],
            'train_lagrangian': [],
            'val_fnr': [],
            'val_fpr': [],
            'val_f1': [],
            'mean_multiplier': [],
            'mean_violation': []
        }
        
        print(f"Initialized FeasibleTrainer:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Device: {device}")
        print(f"  FPR weight: {fpr_weight}")
        print(f"  FPR temp: {fpr_temperature}")
        print(f"  Margin: {margin}")
    
    def initialize_multipliers(self, pos_indices: torch.Tensor, n_pos: int):
        """Initialize persistent multipliers with positive sample indices."""
        self.multipliers = PersistentMultipliers(
            n_pos=n_pos,
            pos_indices=pos_indices,
            lr_multiplier=0.01,
            pi_p=0.1,
            pi_i=0.01,
            slack_weight=1.0,
            use_slack=True,
            device=self.device
        )
        print(f"  ✓ Multipliers initialized with {n_pos} positives")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        total_fpr_loss = 0.0
        total_lagrangian = 0.0
        num_batches = 0
        
        all_violations = []
        all_multipliers = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch_idx, batch_data in enumerate(pbar):
            # Dataset yields (view1, view2, label, global_idx); use view1 + global idx.
            images, labels, global_indices = unpack_batch(batch_data, self.device)
            images = images.to(self.device)
            labels = labels.to(self.device)
            global_indices = global_indices.to(self.device)

            # --- FORWARD (autocast for the heavy backbone; geometry/loss in fp32) ---
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                logits = outputs['logits']
            embeddings = outputs['embeddings'].float()

            # --- BALL INFO (detached constants for this batch) ---
            with torch.no_grad():
                assignments = self.union_of_balls.assign_points(embeddings)
                if len(self.union_of_balls) > 0:
                    ball_centers = torch.stack(
                        [ball.center for ball in self.union_of_balls.balls]
                    ).float()
                    ball_radii = torch.tensor(
                        [ball.radius for ball in self.union_of_balls.balls],
                        device=self.device, dtype=torch.float32
                    )
                else:
                    ball_centers = torch.zeros(1, embeddings.shape[1], device=self.device)
                    ball_radii = torch.tensor([1.0], device=self.device)

            # --- CONSTRAINT VIOLATIONS (POSITIVES) ---
            pos_mask = labels == 1
            pos_embeddings = embeddings[pos_mask]
            pos_global_indices = global_indices[pos_mask]

            if len(pos_embeddings) > 0:
                pos_assignments = assignments[pos_mask]
                assigned_centers = ball_centers[pos_assignments]
                assigned_radii = ball_radii[pos_assignments]

                diff = pos_embeddings - assigned_centers
                dist_sq = torch.sum(diff ** 2, dim=1)
                max_dist_sq = (assigned_radii - self.margin).clamp(min=0.0) ** 2
                violation = F.relu(dist_sq - max_dist_sq)

                if self.multipliers is not None:
                    λ_batch = self.multipliers.get_multipliers(pos_global_indices)
                    λ_mean = λ_batch.mean().item()
                else:
                    λ_batch = torch.zeros_like(violation)
                    λ_mean = 0.0

                lagrangian_term = (λ_batch * violation).mean()
            else:
                violation = torch.tensor(0.0, device=self.device)
                lagrangian_term = torch.tensor(0.0, device=self.device)
                λ_mean = 0.0

            # --- FPR LOSS (NEGATIVES) — push negatives at least `margin` OUTSIDE ---
            neg_mask = labels == 0
            if neg_mask.any():
                neg_embeddings = embeddings[neg_mask]
                signed_dists = self.union_of_balls.distance_to_nearest(neg_embeddings)
                # signed_dist > 0 => inside; +margin enforces a clearance band outside
                fpr_loss = (F.relu(signed_dists + self.margin) ** 2).mean()
            else:
                fpr_loss = torch.tensor(0.0, device=self.device)

            # --- AUXILIARY BCE (classifier head, fp32) ---
            bce_loss = F.binary_cross_entropy_with_logits(
                logits.float().squeeze(-1), labels.float()
            )

            # --- TOTAL LOSS ---
            total_loss_val = self.fpr_weight * fpr_loss + lagrangian_term + 0.1 * bce_loss

            # --- DETACH VALUES FOR LOGGING (BEFORE BACKWARD) ---
            loss_value = total_loss_val.detach().item()
            fpr_value = fpr_loss.detach().item()
            lag_value = lagrangian_term.detach().item() if isinstance(lagrangian_term, torch.Tensor) else 0.0

            # --- BACKWARD (PRIMAL STEP, AMP-scaled) ---
            self.scaler.scale(total_loss_val).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # --- UPDATE MULTIPLIERS (DUAL STEP) with DETACHED violation ---
            if len(pos_embeddings) > 0 and self.multipliers is not None:
                self.multipliers.update(pos_global_indices, violation.detach(), epoch)

            # --- TRACK METRICS ---
            total_loss += loss_value
            total_fpr_loss += fpr_value
            total_lagrangian += lag_value
            num_batches += 1

            if len(pos_embeddings) > 0:
                all_violations.append(violation.mean().item())
                if self.multipliers is not None:
                    stats = self.multipliers.get_stats()
                    all_multipliers.append(stats['mean'])

            pbar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'fpr': f'{fpr_value:.4f}',
                'λ': f'{λ_mean:.4f}'
            })

        # --- RECOMPUTE BALLS on CLEAN, POSITIVE-ONLY embeddings ---
        # CRITICAL: the union encloses POSITIVES. Refitting on all points (incl.
        # negatives) lets the radius balloon to cover negatives -> FPR -> 1.
        # Also use the deterministic eval_loader so the geometry isn't fit on
        # random augmentations.
        all_embeddings = []
        all_labels_full = []
        with torch.no_grad():
            for batch_data in self.eval_loader:
                images, labels, _ = unpack_batch(batch_data, self.device)
                images = images.to(self.device)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                all_embeddings.append(outputs['embeddings'].float())
                all_labels_full.append(labels.to(self.device))

        if all_embeddings:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            all_labels_full = torch.cat(all_labels_full, dim=0)
            pos_only = all_embeddings[all_labels_full == 1]
            neg_only = all_embeddings[all_labels_full == 0]
            if len(pos_only) > 0:
                self.union_of_balls.recompute(
                    pos_only,
                    method='quantile',  # quantile radius = training pressure
                    update_centers=True,
                    update_radii=True
                )
                # CAP each ball below the negatives (robust low quantile) so a spread
                # positive cluster can't balloon and engulf negatives -> keeps val FPR
                # meaningful. Matches the finalize sizing rule.
                if len(neg_only) > 0:
                    for ball in self.union_of_balls.balls:
                        nd = torch.norm(neg_only - ball.center, dim=1)
                        cap = (torch.quantile(nd, 0.01) - self.union_of_balls.margin).item()
                        ball.radius = max(min(ball.radius, cap), 1e-6)

        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'fpr_loss': total_fpr_loss / num_batches if num_batches > 0 else 0.0,
            'lagrangian': total_lagrangian / num_batches if num_batches > 0 else 0.0,
            'mean_violation': np.mean(all_violations) if all_violations else 0,
            'mean_multiplier': np.mean(all_multipliers) if all_multipliers else 0
        }

    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        
        all_labels = []
        all_scores = []
        all_preds = []
        all_abstained = []
        all_embeddings = []
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                images, labels, _ = unpack_batch(batch_data, self.device)
                images = images.to(self.device)
                labels = labels.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                embeddings = outputs['embeddings'].float()
                logits = outputs['logits']

                scores = torch.sigmoid(logits).squeeze()

                # Same abstain-by-default rule as predict():
                #   inside positive ball -> POS; in negative cloud -> NEG; else ABSTAIN.
                signed = self.union_of_balls.distance_to_nearest(embeddings)  # R - d
                inside = signed >= 0        # d <= R -> positive
                neg_ref = getattr(self.model, '_train_neg_embeddings', None)
                neg_thr = getattr(self.model, '_neg_threshold', None)
                if neg_ref is not None and len(neg_ref) > 0 and neg_thr is not None:
                    d_neg = torch.cdist(embeddings, neg_ref).min(dim=1).values
                    conf_neg = d_neg <= neg_thr
                else:
                    conf_neg = signed < -self.union_of_balls.margin
                predict_negative = conf_neg & (~inside)
                preds = inside.float()
                abstained = (~inside) & (~predict_negative)

                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_abstained.extend(abstained.cpu().numpy())
                all_embeddings.extend(embeddings.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        all_preds = np.array(all_preds)
        all_abstained = np.array(all_abstained)
        
        metrics = compute_all_metrics(all_labels, all_scores, all_preds, all_abstained)
        
        # Add geometric metrics
        if len(self.union_of_balls) > 0:
            radii = [ball.radius for ball in self.union_of_balls.balls]
            metrics['num_balls'] = len(radii)
            metrics['radius_mean'] = np.mean(radii)
            metrics['radius_std'] = np.std(radii)
            metrics['radius_min'] = np.min(radii)
            metrics['radius_max'] = np.max(radii)
        
        return metrics
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("STARTING FEASIBLE LEARNING TRAINING")
        print("="*60)
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Balls: {len(self.union_of_balls)}")
        print(f"Multipliers: {'✓' if self.multipliers else '✗'}")
        print("="*60 + "\n")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch(epoch)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  FPR Loss: {train_metrics['fpr_loss']:.4f}")
            print(f"  Lagrangian: {train_metrics['lagrangian']:.4f}")
            print(f"  Mean λ: {train_metrics['mean_multiplier']:.4f}")
            print(f"  LR: {current_lr:.8f}")
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_fpr_loss'].append(train_metrics['fpr_loss'])
            self.history['train_lagrangian'].append(train_metrics['lagrangian'])
            self.history['mean_multiplier'].append(train_metrics['mean_multiplier'])
            self.history['mean_violation'].append(train_metrics['mean_violation'])
            
            # Validate periodically
            if (epoch + 1) % self.eval_freq == 0 or (epoch + 1) == self.num_epochs:
                val_metrics = self.validate()

                # Raw (over ALL samples) — these lump abstained samples in.
                fnr = val_metrics.get('fnr', 1.0)
                fpr = val_metrics.get('fpr', 1.0)
                f1 = val_metrics.get('f1', 0.0)
                abstention = val_metrics.get('abstention_rate', 0.0)
                # Selective (over the COVERED, non-abstained samples) — the right
                # numbers for an abstention system. Abstained = deferred to human.
                sel_fnr = val_metrics.get('non_abstained_fnr', fnr)
                sel_fpr = val_metrics.get('non_abstained_fpr', fpr)
                coverage = 1.0 - abstention

                print(f"  Val FNR (raw): {fnr:.6f}   FPR (raw): {fpr:.6f}   F1: {f1:.6f}")
                print(f"  Val selective  FNR: {sel_fnr:.6f}  FPR: {sel_fpr:.6f}  "
                      f"(coverage {coverage:.3f}, abstain {abstention:.3f})")
                print(f"  Balls: {len(self.union_of_balls)}")

                self.history['val_fnr'].append(fnr)
                self.history['val_fpr'].append(fpr)
                self.history['val_f1'].append(f1)

                # --- Best-model selection on SELECTIVE error (abstained excluded,
                # per the design doc), with a small abstention penalty so we don't
                # reward a model that just abstains on everything.
                score = sel_fnr + sel_fpr + 0.1 * abstention
                if score < getattr(self, 'best_score', float('inf')):
                    self.best_score = score
                    self.best_val_fnr = sel_fnr
                    self.best_val_fpr = sel_fpr
                    self.best_epoch = epoch
                    print(f"  ★ New best! selective FNR={sel_fnr:.6f}, FPR={sel_fpr:.6f}, "
                          f"abstain={abstention:.3f} (score={score:.6f})")
                    if self.save_dir:
                        self.save_checkpoint("best_model.pt")

                if self.wandb:
                    self.wandb.log_validation(epoch, val_metrics)
            
            if self.save_dir and (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"epoch_{epoch+1:03d}.pt")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Best epoch: {self.best_epoch+1}")
        print(f"Best Val FNR: {self.best_val_fnr:.6f}")
        print(f"Best Val FPR: {self.best_val_fpr:.6f}")
        print(f"Final balls: {len(self.union_of_balls)}")
        print("="*60)
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        if self.save_dir is None:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'union_of_balls': self.union_of_balls.to_dict(),
            'best_val_fnr': self.best_val_fnr,
            'best_val_fpr': self.best_val_fpr,
            'best_epoch': self.best_epoch,
            'history': self.history
        }
        
        # Save multipliers state if available
        if self.multipliers is not None:
            checkpoint['multipliers'] = {
                'lambdas': self.multipliers.lambdas.cpu(),
                'integrals': self.multipliers.integrals.cpu(),
                'prev_g': self.multipliers.prev_g.cpu(),
                'pos_indices': self.multipliers.pos_indices.cpu()
            }
        
        torch.save(checkpoint, path)
        print(f"  💾 Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_fnr = checkpoint.get('best_val_fnr', 1.0)
        self.best_val_fpr = checkpoint.get('best_val_fpr', 1.0)
        self.best_epoch = checkpoint.get('best_epoch', -1)
        self.history = checkpoint.get('history', {})
        
        # Restore multipliers if present
        if 'multipliers' in checkpoint and self.multipliers is not None:
            m = checkpoint['multipliers']
            self.multipliers.lambdas = m['lambdas'].to(self.device)
            self.multipliers.integrals = m['integrals'].to(self.device)
            self.multipliers.prev_g = m['prev_g'].to(self.device)
            print(f"  ✓ Restored multipliers state")
        
        print(f"  📥 Checkpoint loaded: {path}")
        print(f"  Resuming at epoch: {self.current_epoch}")
        print(f"  Best FNR: {self.best_val_fnr:.6f}")
        print(f"  Best FPR: {self.best_val_fpr:.6f}")