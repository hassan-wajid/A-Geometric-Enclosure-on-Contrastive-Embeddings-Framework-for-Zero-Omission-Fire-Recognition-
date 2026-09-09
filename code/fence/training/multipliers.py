"""
Persistent per-sample Lagrange multipliers for Feasible Learning.

One multiplier λᵢ per training positive, indexed by global sample id.
Implements νPI controller with resilient slack.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np


class PersistentMultipliers:
    """
    One Lagrange multiplier λᵢ per training positive, indexed by global sample id.
    
    Implements νPI controller (Sohrabi et al. 2024) with resilient slack.
    """
    
    def __init__(self, 
                 n_pos: int,
                 pos_indices: torch.Tensor,
                 lr_multiplier: float = 0.01,
                 pi_p: float = 0.1,
                 pi_i: float = 0.01,
                 slack_weight: float = 1.0,
                 use_slack: bool = True,
                 device: str = 'cpu'):
        """
        Args:
            n_pos: Total number of positive samples
            pos_indices: Global indices of positive samples (N_pos,)
            lr_multiplier: Learning rate for multipliers (η)
            pi_p: Proportional gain
            pi_i: Integral gain
            slack_weight: α for resilient slack (u_i = λ_i / α)
            use_slack: Whether to use resilient slack
            device: Device to use
        """
        self.n_pos = n_pos
        self.lr_multiplier = lr_multiplier
        self.pi_p = pi_p
        self.pi_i = pi_i
        self.slack_weight = slack_weight
        self.use_slack = use_slack
        self.device = device
        
        # Persistent state: one λ per positive
        self.lambdas = torch.zeros(n_pos, device=device)
        self.integrals = torch.zeros(n_pos, device=device)
        self.prev_g = torch.zeros(n_pos, device=device)
        
        # Map global index to position in buffer
        self.pos_indices = pos_indices.to(device)
        self.idx_to_pos = {int(idx): i for i, idx in enumerate(pos_indices.cpu().numpy())}
        
        # History for logging
        self.history = []
        
        print(f"Initialized PersistentMultipliers:")
        print(f"  N positives: {n_pos}")
        print(f"  LR: {lr_multiplier}, PI: ({pi_p}, {pi_i})")
        print(f"  Slack: {use_slack}, Weight: {slack_weight}")
    
    def get_multipliers(self, global_indices: torch.Tensor) -> torch.Tensor:
        """
        Get multipliers for specific global indices.
        
        Args:
            global_indices: Global indices of samples (B,)
            
        Returns:
            Multipliers λ for those samples (B,)
        """
        if len(global_indices) == 0:
            return torch.tensor([], device=self.device)

        # Positives this object never tracked (e.g. val/cal positives when
        # finalizing on 'all') get multiplier 0 — they're simply not outliers.
        out = torch.zeros(len(global_indices), device=self.device)
        for i, idx in enumerate(global_indices.cpu()):
            pos = self.idx_to_pos.get(int(idx.item()))
            if pos is not None:
                out[i] = self.lambdas[pos]
        return out
    
    def update(self, 
               global_indices: torch.Tensor, 
               violations: torch.Tensor,
               epoch: int = 0) -> torch.Tensor:
        """
        Update multipliers using νPI controller.
        
        λ_i ← max(0, λ_i + η * (P * effective_g + I * integral_i))
        effective_g = g_i - λ_i/α (resilient slack)
        
        Args:
            global_indices: Global indices of samples (B,)
            violations: Constraint violations g_i for those samples (B,)
            epoch: Current epoch
            
        Returns:
            Updated multipliers for those samples (B,)
        """
        if len(global_indices) == 0:
            return self.lambdas
        
        positions = torch.tensor(
            [self.idx_to_pos[int(idx.item())] for idx in global_indices.cpu()],
            device=self.device
        )
        
        # Get current values
        current_λ = self.lambdas[positions]
        current_integral = self.integrals[positions]
        
        # Compute effective violation (resilient slack)
        if self.use_slack:
            slack = current_λ / self.slack_weight
            effective_g = violations - slack
        else:
            effective_g = violations
        
        # Update integral
        self.integrals[positions] = current_integral + effective_g
        
        # PI update
        delta = (self.pi_p * effective_g + 
                 self.pi_i * self.integrals[positions])
        
        # Apply update and project to non-negative
        new_λ = F.relu(current_λ + self.lr_multiplier * delta)
        
        # Store updated values
        self.lambdas[positions] = new_λ
        self.prev_g[positions] = effective_g
        
        # Log history
        if epoch % 5 == 0:
            self.history.append({
                'epoch': epoch,
                'mean_λ': self.lambdas.mean().item(),
                'max_λ': self.lambdas.max().item(),
                'mean_violation': violations.mean().item(),
                'max_violation': violations.max().item()
            })
        
        return new_λ
    
    def get_slack(self, global_indices: torch.Tensor) -> torch.Tensor:
        """Get slack u_i = λ_i / α for samples."""
        if not self.use_slack:
            return torch.zeros_like(global_indices, dtype=torch.float32, device=self.device)
        λ = self.get_multipliers(global_indices)
        return λ / self.slack_weight
    
    def get_all_slack(self) -> torch.Tensor:
        """Get slack for all positives."""
        if not self.use_slack:
            return torch.zeros(self.n_pos, device=self.device)
        return self.lambdas / self.slack_weight
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistics about multipliers."""
        if self.n_pos == 0:
            return {'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0}
        
        return {
            'mean': self.lambdas.mean().item(),
            'max': self.lambdas.max().item(),
            'min': self.lambdas.min().item(),
            'std': self.lambdas.std().item()
        }
    
    def get_anomaly_scores(self) -> torch.Tensor:
        """Get anomaly scores (slack) for all positives."""
        return self.get_all_slack()
    
    def reset(self):
        """Reset all multipliers to zero."""
        self.lambdas.zero_()
        self.integrals.zero_()
        self.prev_g.zero_()