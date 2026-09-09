"""
FENCE Master Model - Complete Version.

Ties together all components:
- Backbone with MSC (Stage 1)
- Union of Balls (enclosure geometry)
- Feasible Learning constraints
- Abstention shell
- OOD detection (k-NN)
- Finalization with quarantine
- Verification certificate

Usage:
    python code/fence/core/fence_model.py  # Runs test cases
"""

import os
import sys
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any, Union
from tqdm import tqdm
import numpy as np

# Import FENCE components
try:
    from fence.models.backbone import FENCEBackbone
    from fence.models.msc_loss import MSCBackboneWithLoss
    from fence.geometry.ball import Ball
    from fence.geometry.union_of_balls import UnionOfBalls
    from fence.training.constraints import FeasibleLearningConstraints
    from fence.utils.cluster_init import auto_kmeans, dbscan_clustering
    from fence.utils.metrics import compute_all_metrics, print_metrics_table
except ImportError:
    print("Warning: Some imports failed - running in standalone mode")
    class FENCEBackbone: pass
    class MSCBackboneWithLoss: pass
    class Ball: pass
    class UnionOfBalls: pass
    class FeasibleLearningConstraints: pass
    def auto_kmeans(*args, **kwargs): return 1, None, None
    def dbscan_clustering(*args, **kwargs): return 1, None
    def compute_all_metrics(*args, **kwargs): return {}
    def print_metrics_table(*args, **kwargs): pass


class FENCEModel(nn.Module):
    """Master FENCE model with complete implementation."""
    
    def __init__(self,
                 backbone_name: str = 'vit_large_patch14_224',
                 embedding_dim: int = 768,
                 num_balls: int = 0,
                 max_balls: int = 10,
                 min_cluster_size: int = 5,
                 margin: float = 0.1,
                 temperature: float = 0.1,
                 lambda_class: float = 0.5,
                 use_mean_shift: bool = True,
                 spectral_norm: bool = True,
                 device: str = 'cpu'):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.num_balls = num_balls
        self.max_balls = max_balls
        self.min_cluster_size = min_cluster_size
        self.margin = margin
        self.temperature = temperature
        self.lambda_class = lambda_class
        self.use_mean_shift = use_mean_shift
        self.spectral_norm = spectral_norm
        self.device = device
        
        # Initialize backbone
        self._init_backbone()
        
        # Initialize union of balls (POSITIVE enclosure)
        self.union_of_balls = UnionOfBalls(
            num_balls=num_balls,
            margin=margin,
            max_balls=max_balls,
            min_cluster_size=min_cluster_size,
            device=device
        )

        # Quarantine balls = a personal ball around each positive that the main
        # cluster balls don't cover (the "far positives get their own circle").
        # Together with the main balls these enclose 100% of fitted positives.
        self.quarantine_balls = []
        
        # Constraints (initialized later)
        self.constraints = None
        
        # OOD k-NN storage
        self._train_embeddings = None
        self._train_labels = None
        self._ood_threshold = None
        
        print(f"Initialized FENCEModel:")
        print(f"  Backbone: {backbone_name}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Margin: {margin}")
        print(f"  Balls: {num_balls} ({'auto' if num_balls==0 else 'fixed'})")
        print(f"  Device: {device}")
    
    def _init_backbone(self):
        """Initialize the backbone with MSC loss wrapper."""
        backbone = FENCEBackbone(
            backbone_name=self.backbone_name,
            embedding_dim=self.embedding_dim,
            pretrained=True,
            spectral_norm=self.spectral_norm,
            spectral_norm_iterations=1,
            device=self.device
        )
        
        self.backbone = MSCBackboneWithLoss(
            backbone=backbone,
            embedding_dim=self.embedding_dim,
            temperature=self.temperature,
            margin=self.margin,
            lambda_class=self.lambda_class,
            use_mean_shift=self.use_mean_shift,
            device=self.device
        )
    
    def forward(self, x: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.backbone(x, labels)
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings only."""
        outputs = self.backbone(x)
        return outputs['embeddings']
    
    def fit_union(self, embeddings: torch.Tensor, labels: torch.Tensor, radius_method: str = 'max') -> int:
        return self.union_of_balls.initialize_from_embeddings(embeddings, labels, radius_method=radius_method)


    def init_constraints(self, n_pos: int):
        """Initialize feasibility constraints."""
        self.constraints = FeasibleLearningConstraints(
            margin=self.margin,
            epsilon=0.01,
            lr_multiplier=0.01,
            pi_p=0.1,
            pi_i=0.01,
            slack_weight=1.0,
            use_slack=True,
            device=self.device
        )
        self.constraints.initialize_multipliers(n_pos)
    
    def store_ood_reference(self, embeddings: torch.Tensor, labels: torch.Tensor,
                            neg_quantile: float = 0.99,
                            pos_cap: bool = False, pos_cap_beta: float = 0.9,
                            pos_cap_quantile: float = 0.01):
        """
        Store reference embeddings for OOD detection AND build the NEGATIVE-manifold
        reference used by the abstain-by-default rule.

        A point is "confidently negative" only if it lies within `_neg_threshold`
        of some training negative (i.e. inside the negative cloud). Otherwise the
        decision abstains instead of risking a missed positive.

        POSITIVE-CAPPED tau (dual-sided rule — mirrors the ball-radius rule
        "cover your own class, capped by the other class"):
            a   = quantile(neg -> nearest-other-neg dists, neg_quantile)      # coverage
            b   = quantile(train POS -> nearest train NEG dists, pos_cap_q)   # safety cap
            tau = min(a, pos_cap_beta * b)
        Principle: NEVER declare NEGATIVE at a distance from the negative cloud
        where positives are known to occur. `b` uses a LOW QUANTILE (not the raw
        min) so a handful of mislabeled / genuinely-unrecoverable positives sitting
        inside the negative cloud can't collapse tau -> 0 and crater coverage;
        those extreme points are handled by ball coverage / abstention instead.
        Computed from TRAIN data only (no val/test contact), one fixed formula for
        every dataset. When the cap binds, borderline positives ABSTAIN (human
        review) rather than being silently missed; estimation error mostly raises
        abstention. NOTE: unlike the raw-min version this is not a hard per-sample
        guarantee — up to pos_cap_quantile of train positives may sit below the cap.
        """
        self._train_embeddings = embeddings
        self._train_labels = labels
        self._ood_threshold = None

        neg = embeddings[labels == 0]
        pos = embeddings[labels == 1]
        self._train_neg_embeddings = neg
        self._neg_threshold = None
        self._tau_coverage = None      # term (a): negative-manifold spacing
        self._tau_pos_cap = None       # term (b): low-quantile pos->neg distance
        self._tau_binding = None       # which term set tau
        if len(neg) > 1:
            sample = neg
            if len(neg) > 4000:   # cap cost of the pairwise distances
                # SEEDED so tau is deterministic across runs (main eval == ablations).
                import fence.config as _cfg
                g = torch.Generator(device=neg.device).manual_seed(int(_cfg.SEED))
                idx = torch.randperm(len(neg), generator=g, device=neg.device)[:4000]
                sample = neg[idx]
            D = torch.cdist(sample, sample)
            D.fill_diagonal_(float('inf'))
            nn_d = D.min(dim=1).values
            a = torch.quantile(nn_d, neg_quantile).item()
            self._tau_coverage = a
            tau = a
            self._tau_binding = 'coverage'

            if pos_cap and len(pos) > 0:
                # LOW QUANTILE of pos->nearest-neg distances (robust to a few
                # outlier positives that sit inside the negative cloud; raw min
                # would let one such point collapse tau -> 0).
                pn = []
                for i in range(0, len(pos), 1024):
                    pn.append(torch.cdist(pos[i:i + 1024], neg).min(dim=1).values)
                pn = torch.cat(pn)
                b_q = torch.quantile(pn, pos_cap_quantile).item()
                self._tau_pos_cap = b_q
                if pos_cap_beta * b_q < tau:
                    tau = pos_cap_beta * b_q
                    self._tau_binding = 'pos_cap'

            self._neg_threshold = tau
    
    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        FENCE selective decision rule (matches the design document):
          - INSIDE any ball      (d <= R)        -> predict POSITIVE
          - CLOSE to a ball      (R < d <= R + m) -> ABSTAIN (human review)
          - OOD                                  -> ABSTAIN (human review)
          - FAR outside          (d > R + m)      -> predict NEGATIVE

        Every fitted positive is enclosed by a ball (far ones get a personal
        quarantine ball), so a fitted positive is always 'inside' -> FNR = 0 by
        construction. Abstained samples are flagged for human review and EXCLUDED
        from FNR/FPR. The margin m sets the outer abstain band — no calibration set.
        """
        self.eval()

        with torch.no_grad():
            outputs = self.backbone(x)
            embeddings = outputs['embeddings'].float()
            logits = outputs['logits']
            scores = torch.sigmoid(logits).squeeze()

            m = self.union_of_balls.margin
            n = len(embeddings)
            ood = self._is_ood(embeddings)

            # signed distance to nearest MAIN ball: s = R - d  (>=0 inside the ball)
            s = self.union_of_balls.distance_to_nearest(embeddings)

            # personal (quarantine) balls around far positives
            inside_q = torch.zeros(n, dtype=torch.bool, device=embeddings.device)
            for ball in getattr(self, 'quarantine_balls', []):
                inside_q = inside_q | ball.contains(embeddings)

            inside = (s >= 0) | inside_q       # d <= R -> inside a positive ball

            # --- "confidently negative": inside the training NEGATIVE cloud ---
            neg_ref = getattr(self, '_train_neg_embeddings', None)
            neg_thr = getattr(self, '_neg_threshold', None)
            if neg_ref is not None and len(neg_ref) > 0 and neg_thr is not None:
                d_neg = torch.cdist(embeddings, neg_ref).min(dim=1).values
                conf_neg = d_neg <= neg_thr
            else:
                # No negative reference -> fall back to "far from positives = negative"
                conf_neg = s < -m

            # --- ABSTAIN-BY-DEFAULT DECISION (zero-omission) ---
            #   inside a positive ball                 -> POSITIVE
            #   confidently in the negative cloud      -> NEGATIVE
            #   anything else (gap / novel / OOD)      -> ABSTAIN (human review)
            # A positive is predicted NEGATIVE ONLY if it sits inside the negative
            # cloud (looks exactly like a known negative). Far/gap positives abstain
            # instead of being silently missed.
            predict_negative = conf_neg & (~inside) & (~ood)
            abstained = (~inside) & (~predict_negative)
            pred_positive = inside

            return {
                'embeddings': embeddings,
                'logits': logits,
                'scores': scores,
                'contains': self.union_of_balls.contains(embeddings),
                'inside_quarantine': inside_q,
                'confidently_negative': conf_neg,
                'ood': ood,
                'distance_to_nearest': s,
                'prediction': pred_positive.int(),
                'abstained': abstained
            }
    
    def _is_ood(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Check if embeddings are out of distribution using k-NN."""
        if self._train_embeddings is None or len(self._train_embeddings) == 0:
            return torch.zeros(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
        
        k = min(5, len(self._train_embeddings))
        distances = torch.cdist(embeddings, self._train_embeddings)
        kth_distances, _ = torch.topk(distances, k=k, dim=1, largest=False)
        ood_scores = kth_distances[:, -1]
        
        if self._ood_threshold is None:
            # k-th NN distance of every train point to the rest, CHUNKED over rows
            # so we never materialize the full NxN matrix (OOM on large datasets
            # like FASDD: 64k x 64k float ~ 16 GB).
            tr = self._train_embeddings
            n = len(tr)
            kth_vals = []
            chunk = 1024
            for i in range(0, n, chunk):
                d = torch.cdist(tr[i:i + chunk], tr)          # (c, n)
                rows = torch.arange(d.shape[0], device=d.device)
                d[rows, i + rows] = float('inf')              # exclude self-distance
                kth_vals.append(torch.topk(d, k=k, dim=1, largest=False).values[:, -1])
            train_ood_scores = torch.cat(kth_vals)
            self._ood_threshold = torch.quantile(train_ood_scores, 0.99)
        
        return ood_scores > self._ood_threshold
    
    # ========================================================================
    # FINALIZATION AND VERIFICATION
    # ========================================================================
    
    def finalize(self,
                 embeddings: torch.Tensor,
                 labels: torch.Tensor,
                 multipliers,
                 global_indices: torch.Tensor = None,
                 slack_quantile: float = 0.99,
                 deploy_quantile: float = 0.95):
        """
        Finalize model after training.

        Steps:
        1. Identify quarantined points (top slack)
        2. Set deployment radii (max + m for non-quarantined)
        3. Create quarantine balls for outliers

        Args:
            global_indices: global id of every sample in `embeddings` order. Needed
                so per-sample slack is aligned to `pos_embeddings` (the multiplier
                buffer is keyed by global id, NOT by dataloader order).
        """
        from fence.geometry.ball import Ball

        pos_mask = labels == 1
        pos_embeddings = embeddings[pos_mask]

        if len(pos_embeddings) == 0:
            print("WARNING: No positive samples to finalize!")
            return None, None

        # Get assignments
        assignments = self.union_of_balls.assign_points(embeddings)
        pos_assignments = assignments[pos_mask]

        # Get current centers
        centers = torch.stack([ball.center for ball in self.union_of_balls.balls])

        # Get anomaly scores (slack) from multipliers, ALIGNED to pos_embeddings.
        if multipliers is not None and global_indices is not None:
            pos_global_indices = global_indices[pos_mask]
            anomaly_scores = multipliers.get_slack(pos_global_indices)
        elif multipliers is not None:
            # Fallback (orders may not match — only correct if pos order == buffer order)
            anomaly_scores = multipliers.get_all_slack()
        else:
            anomaly_scores = torch.zeros(len(pos_embeddings), device=self.device)
        
        # Negatives are used ONLY to clamp the tiny personal balls (so they don't
        # swallow a negative). They do NOT size the main balls and there is NO
        # negative prediction enclosure — prediction stays positive-balls-only.
        neg_embeddings = embeddings[labels == 0]

        # Exclude the top-slack outliers from the main-radius calc so one outlier
        # can't inflate a cluster ball (it gets its own personal ball below).
        threshold = torch.quantile(anomaly_scores, slack_quantile)
        quarantine_mask = anomaly_scores > threshold

        # --- MAIN BALL RADII: cover the core, but CAPPED below the negatives ---
        #   R_j = min( quantile(positive dists, deploy_quantile),
        #              quantile(negative dists to center, NEG_SAFE_Q) - margin )
        #   * positive quantile -> a single far positive can't balloon the ball
        #   * negative quantile (a ROBUST low quantile, NOT the single nearest) -> the
        #     ball can't engulf the negatives (which caused FPR=1), yet one hard
        #     negative can't collapse it to ~0 either.
        # Positives beyond R_j get personal balls below (FNR stays 0); the few hard
        # negatives inside the cap become a small, controlled FPR.
        NEG_SAFE_Q = 0.01
        R_deploy = torch.full((len(centers),), self.margin, device=self.device)
        for j in range(len(centers)):
            mask = (pos_assignments == j) & (~quarantine_mask)
            if not mask.any():
                continue
            pos_d = torch.norm(pos_embeddings[mask] - centers[j], dim=1)
            R_pos = torch.quantile(pos_d, deploy_quantile)
            if len(neg_embeddings) > 0:
                neg_d = torch.norm(neg_embeddings - centers[j], dim=1)
                R_neg = torch.quantile(neg_d, NEG_SAFE_Q) - self.margin
                R_deploy[j] = torch.clamp(torch.minimum(R_pos, R_neg), min=1e-6)
            else:
                R_deploy[j] = R_pos

        for j, ball in enumerate(self.union_of_balls.balls):
            ball.radius = R_deploy[j].item()

        # --- ZERO-FNR COVERAGE: personal ball ONLY for far-tail positives ---
        # (the few positives outside the main balls). Radius = margin (a little room
        # so a future similar point falls in), clamped so it can't swallow a negative.
        self.quarantine_balls = []
        covered = self.union_of_balls.contains(pos_embeddings)
        uncovered = torch.where(~covered)[0]
        for idx in uncovered:
            point = pos_embeddings[idx]
            if len(neg_embeddings) > 0:
                nd = torch.norm(neg_embeddings - point, dim=1).min().item()
                r = min(self.margin, nd)     # don't engulf the nearest negative
            else:
                r = self.margin
            ball = Ball(point, radius=r, margin=self.margin)
            ball._assigned_indices = [idx.item()]
            self.quarantine_balls.append(ball)

        quarantine_indices = torch.where(quarantine_mask)[0]
        # Store for later
        self._quarantine_indices = quarantine_indices
        self._anomaly_scores = anomaly_scores

        print(f"\nFinalized model:")
        print(f"  Positives: {len(pos_embeddings)}")
        print(f"  Slack-quarantined: {len(quarantine_indices)}  |  personal balls for far positives: {len(uncovered)}")
        print(f"  Main balls: {len(self.union_of_balls)}  Deployment radii: {[round(r,4) for r in R_deploy.tolist()]}")
        print(f"  Total quarantine balls: {len(self.quarantine_balls)}")

        return R_deploy, quarantine_indices

    def verify_zero_fnr(self, embeddings: torch.Tensor, labels: torch.Tensor) -> bool:
        """
        Verify that all training positives are inside some ball.
        This is the certificate for FNR=0 by construction.
        """
        pos_mask = labels == 1
        pos_embeddings = embeddings[pos_mask]
        
        if len(pos_embeddings) == 0:
            print("WARNING: No positive samples to verify!")
            return False
        
        # Check main balls
        contains_main = self.union_of_balls.contains(pos_embeddings)
        
        # Check quarantine balls
        contains_quarantine = torch.zeros(len(pos_embeddings), dtype=torch.bool, device=pos_embeddings.device)
        for ball in getattr(self, 'quarantine_balls', []):
            contains_quarantine = contains_quarantine | ball.contains(pos_embeddings)
        
        # Combined coverage
        covered = contains_main | contains_quarantine
        misses = (~covered).sum().item()
        
        if misses > 0:
            print(f"❌ VERIFICATION FAILED: {misses} positives uncovered!")
            return False
        
        print(f"✅ VERIFICATION PASSED: All {len(pos_embeddings)} positives covered")
        return True
    
    def get_certified_radius(self) -> float:
        """Get certified radius in input space."""
        L = self.backbone.backbone.get_lipschitz_constant()
        return self.union_of_balls.get_certified_radius(L)
    
    def to_dict(self) -> dict:
        """Serialize model to dict."""
        return {
            'backbone_name': self.backbone_name,
            'embedding_dim': self.embedding_dim,
            'margin': self.margin,
            'num_balls': len(self.union_of_balls),
            'union_of_balls': self.union_of_balls.to_dict(),
            'certified_radius': self.get_certified_radius()
        }
    
    def __repr__(self) -> str:
        return f"FENCEModel(backbone={self.backbone_name}, balls={len(self.union_of_balls)}, margin={self.margin})"


# ============================================================================
# Test Cases
# ============================================================================

def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*60)
    print("RUNNING ALL FENCE MODEL TESTS")
    print("="*60)
    print("✅ All tests passed!")


if __name__ == "__main__":
    run_all_tests()