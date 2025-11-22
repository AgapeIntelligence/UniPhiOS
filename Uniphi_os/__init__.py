"""
UniPhi-OS Core Module
Initializes the UniPhi-OS package and exposes key utilities.
"""

# Engine / Core
from .engine import GenesisGeometry

# Utilities
from .utils import (
    soul_key_hash,
    normalize_vector,
    harmonic_mod_369,
    phantom_scalar,
    fuse_fields,
)

# Lightning (PyTorch Lightning integration)
from .lightning import SovarielLightning

# New math / reflection modules
from .phi52_reflection import reflect_vector_52
from .zero_apotheosis import zero_apotheosis_tensor, zero_apotheosis_scalar

# Optional: package-level metadata
__version__ = "0.1.0"
__author__ = "Evie / @3vi3Aetheris"
