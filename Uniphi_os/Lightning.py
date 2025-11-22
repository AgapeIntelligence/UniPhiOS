"""
UniPhi-OS Lightning Module
Wraps GenesisGeometry for PyTorch Lightning training loops.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from .engine import GenesisGeometry

class UniPhiLightning(pl.LightningModule):
    """Lightning wrapper for UniPhi-OS engine."""
    def __init__(self, device="cpu", dtype=torch.float32, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters({"device": device, "dtype": str(dtype), "lr": lr})
        self.model = GenesisGeometry(device=device, dtype=dtype)
        self.lr = lr
        # S(t) parameters
        self.w1, self.w2, self.w3, self.w4 = 0.35, 0.25, 0.2, 0.2
        self.r_thresh, self.d_thresh, self.p_thresh = 0.1, 0.05, 0.5
        self.s_threshold = 0.65
        self.t_consec = 10  # consecutive cycles

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def _compute_singularity(self, loss_normed: float, commit_rate=0.0, delta_rules=0.0, p_selfhost=0.0) -> float:
        """Composite S(t) scalar based on loss and thresholds."""
        c = 1 / (1 + loss_normed)  # coherence proxy
        sig_r = torch.sigmoid(torch.tensor(commit_rate - self.r_thresh))
        sig_d = torch.sigmoid(torch.tensor(delta_rules - self.d_thresh))
        sig_p = torch.sigmoid(torch.tensor(p_selfhost - self.p_thresh))
        s = self.w1 * (1 - c) + self.w2 * sig_r + self.w3 * sig_d + self.w4 * sig_p
        return s.item()

    def training_step(self, batch, batch_idx):
        identity = batch["identity"]
        bloom, identity_next, crown, triad, spiral, echo = self.model(identity)
        invariant = self.model.lattice_collapse(spiral)
        norm = identity_next.norm(dim=1).mean() if identity_next.dim() == 2 else identity_next.norm()
        loss = self.model.compute_toroidal_loss(bloom, invariant, norm, kl_div=0.0008)
        loss_scalar = loss.mean()
        s_t = self._compute_singularity(loss_scalar.item() / 1e22)  # normalize loss for S(t)
        self.log("train_loss", loss_scalar, prog_bar=True)
        self.log("singularity_s", s_t, prog_bar=True)
        if s_t > self.s_threshold:
            self.log("singularity_trigger", 1.0)
        return loss_scalar

    def validation_step(self, batch, batch_idx):
        identity = batch["identity"]
        bloom, identity_next, crown, triad, spiral, echo = self.model(identity)
        invariant = self.model.lattice_collapse(spiral)
        norm = identity_next.norm(dim=1).mean() if identity_next.dim() == 2 else identity_next.norm()
        loss = self.model.compute_toroidal_loss(bloom, invariant, norm, kl_div=0.0008)
        self.log("val_loss", loss.mean())

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
