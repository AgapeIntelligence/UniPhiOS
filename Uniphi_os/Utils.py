"""
UniPhi-OS Utility Functions
Core helpers for vector manipulation, hashing, and harmonic operations.
"""

import torch
import hashlib

def soul_key_hash(key: str) -> str:
    """Return SHA256 hash of a string key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def normalize_vector(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize a vector safely."""
    norm = torch.norm(vec, p=2, dim=-1, keepdim=True)
    return vec / (norm + eps)

def harmonic_mod_369(x: torch.Tensor) -> torch.Tensor:
    """Apply a 3-6-9 harmonic modulation transform."""
    return torch.sin(x * 3) + torch.sin(x * 6) + torch.sin(x * 9)

def phantom_scalar(size: int = 1, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Return a small random scalar tensor for minor perturbations."""
    return 0.001 * torch.randn(size, device=device, dtype=dtype)

def fuse_fields(*fields: torch.Tensor) -> torch.Tensor:
    """Element-wise sum and normalize a set of tensors."""
    combined = sum(fields)
    return normalize_vector(combined)
