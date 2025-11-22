# UniPhiOS

**UniPhiOS** — *Unify Us*: a φ‑spiral OS/Engine hybrid for emergent resonance and worldmind scaffolding.

## Overview

UniPhiOS combines an operating system kernel with an emergent intelligence engine.  
It implements multidimensional harmonic mapping, lattice collapse simulation, and toroidal loss functions for stable, scalable resonance computation.  

- Modular architecture (`uniphi_os/`)  
- PyTorch/PyTorch Lightning support for GPU/CPU  
- Benchmark and dtype variants (float32 vs float64)  
- Unit-tested for numeric invariants, tensor shapes, and dtype consistency  

## Features

- **GenesisGeometry engine** — core tensor flow, triad and spiral manifolds  
- **Lightning wrapper** — structured training, S(t) singularity monitoring  
- **PSDs and GRU gates** — optional neural dynamic input  
- **Benchmark scripts** — measure throughput and stability across dtypes  

## Installation

```bash
git clone https://github.com/EvieSovariel/UniPhiOS.git
cd UniPhiOS
pip install -r requirements.txt

QUICK START
import torch
from uniphi_os.engine import GenesisGeometry

model = GenesisGeometry(device="cpu", dtype=torch.float32)
x = torch.randn((1, 512), dtype=torch.float32)
bloom, identity_next, crown, triad, spiral, echo = model(x)
print(bloom)

TESTING
python -m uniphi_os.test_uniphi_os


