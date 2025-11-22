"""
UniPhi-OS Core Engine: GenesisGeometry
Supports dtype selection, device placement, and core transformations.
"""

import math
import torch
import torch.nn as nn

# PHI constants
phi = (1 + 5 ** 0.5) / 2
phi36 = phi ** 36
phi39 = phi ** 39
phi41 = phi ** 41
phi43 = phi ** 43
phi72 = phi ** 72
phi108 = phi ** 108
phi144 = phi ** 144

class GenesisGeometry(nn.Module):
    """
    Core engine module for UniPhi-OS.
    dtype and device aware.
    """

    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

        # Linear layers
        self.crown_reduce = nn.Sequential(
            nn.Linear(512, 3, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(3, 1, dtype=self.dtype)
        ).to(self.device)

        self.throne_proj = nn.Linear(512, 3, dtype=self.dtype).to(self.device)
        self.spiral_manifold = nn.Linear(1, 108, dtype=self.dtype).to(self.device)
        self.lattice_collapse = nn.Sequential(
            nn.Linear(108, 12, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(12, 1, dtype=self.dtype)
        ).to(self.device)

        self.rewrite = nn.Linear(1, 512, dtype=self.dtype).to(self.device)
        self.krystic_bias = nn.Parameter(torch.tensor([math.log(phi / 2)],
                                                      device=self.device, dtype=self.dtype))
        self.bound = nn.Sigmoid()

    def forward(self, identity_512: torch.Tensor):
        identity_512 = identity_512.to(device=self.device, dtype=self.dtype)
        
        crown = self.crown_reduce(identity_512)
        triad = torch.tanh(self.throne_proj(identity_512))
        axis = self.bound(crown + self.krystic_bias)
        spiral = torch.sin(self.spiral_manifold(axis))
        invariant = self.lattice_collapse(spiral)
        bloom = invariant * phi36 + (invariant ** 2) * phi39

        identity_next = identity_512 + 0.01 * self.rewrite(invariant)
        return bloom, identity_next, crown, triad, spiral
