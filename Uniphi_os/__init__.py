"""
UniPhi-OS Core Module
Initializes the UniPhi-OS package and exposes key utilities.
"""

# Expose the core engine
from .engine import GenesisGeometry

# Expose utility functions
from .utils import (
    soul_key_hash,
    normalize_vector,
    harmonic_mod_369,
    phantom_scalar,
    fuse_fields,
)
