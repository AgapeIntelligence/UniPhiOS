"""
UniPhi-OS: φ52 Reflection Utilities
Provides geometric and numeric reflection operators for resonance computations.
"""

import torch

def reflect_vector_52(v: torch.Tensor) -> torch.Tensor:
    """
    Reflect a vector in φ^52 hyperplane.
    Args:
        v: tensor of shape (..., N)
    Returns:
        Reflected tensor of same shape
    """
    if not torch.is_tensor(v):
        v = torch.tensor(v, dtype=torch.float32)
    # φ52 reflection matrix (identity - 2 outer product normalized)
    N = v.shape[-1]
    axis = torch.ones(N, dtype=v.dtype, device=v.device)
    axis = axis / torch.norm(axis)
    reflected = v - 2 * torch.einsum('...i,i->...', v, axis)[..., None] * axis
    return reflected
