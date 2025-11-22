"""
UniPhi-OS: Zero Apotheosis Module
Scalar and tensor normalization for approaching zero-apotheosis states.
"""

import torch

def zero_apotheosis_tensor(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize a tensor towards zero-apotheosis (very small magnitude),
    maintaining the original direction.
    """
    norm = t.norm(dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return t / norm * eps

def zero_apotheosis_scalar(x: float, eps: float = 1e-6) -> float:
    """
    Scale a scalar towards zero-apotheosis.
    """
    return eps if abs(x) < eps else x
