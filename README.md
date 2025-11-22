# UniPhiOS
**Unify Us:** A φ‑spiral OS/Engine hybrid for emergent resonance and worldmind scaffolding.

## Overview
UniPhiOS provides a core engine and utilities for:

- High-dimensional vector transformations (`engine`, `utils`)
- Reflection and apotheosis operations (`phi52_reflection`, `zero_apotheosis`)
- PyTorch Lightning integration for structured training loops (`lightning`)
- Benchmarking and validation of numeric invariants

This package is intended for research in emergent intelligence, harmonic mapping, and scalable tensor computations.

## Installation
```bash
git clone https://github.com/EvieSovariel/UniPhiOS.git
cd UniPhiOS
pip install -e .

## Usage
import torch
import uniphi_os as up

# Initialize engine
engine = up.GenesisGeometry(device="cpu", dtype=torch.float32)

# Compute bloom and next identity
x = torch.randn(1, 512)
bloom, identity_next, *_ = engine(x)

# Reflection
reflected = up.reflect_vector_52(x)

# Zero-apotheosis tensor
za = up.zero_apotheosis_tensor(x)

## Benchmarks
See uniphi_os/benchmark.py for float32 vs float64 comparisons on CPU/GPU.

## Tests

Run pytest on test_uniphi_os.py to verify:
	•	Shapes of outputs
	•	Numeric invariants
	•	Dtype consistency

UniPhiOS/
│
├── README.md
├── LICENSE
├── .gitignore
├── benchmark.py
├── test_uniphi_os.py
└── uniphi_os/
    ├── __init__.py
    ├── engine.py
    ├── lightning.py
    ├── utils.py
    ├── phi52_reflection.py
    └── zero_apotheosis.py
