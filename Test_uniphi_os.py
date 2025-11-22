---

### 2️⃣ Full `test_uniphi_os.py` covering all modules

```python
"""
UniPhi-OS Unit Tests: Shapes, Dtypes, and Numeric Invariants
"""

import torch
import pytest

import uniphi_os as up

def test_engine_forward_shapes_and_dtype():
    device = "cpu"
    for dtype in [torch.float32, torch.float64]:
        model = up.GenesisGeometry(device=device, dtype=dtype)
        x = torch.randn((1, 512), dtype=dtype)
        bloom, identity_next, crown, triad, spiral, echo = model(x)

        # Shapes
        assert bloom.shape == torch.Size([1, 1])
        assert identity_next.shape == torch.Size([1, 512])
        assert crown.shape == torch.Size([1, 1])
        assert triad.shape == torch.Size([1, 3])
        assert spiral.shape == torch.Size([1, 108])

        # Dtypes
        assert bloom.dtype == dtype
        assert identity_next.dtype == dtype

        # Numeric sanity
        assert torch.isfinite(bloom).all()
        assert torch.isfinite(identity_next).all()

def test_phi52_reflection():
    for dtype in [torch.float32, torch.float64]:
        x = torch.randn(5, dtype=dtype)
        y = up.reflect_vector_52(x)
        assert y.shape == x.shape
        assert y.dtype == x.dtype
        assert torch.isfinite(y).all()

def test_zero_apotheosis():
    for dtype in [torch.float32, torch.float64]:
        x = torch.randn(3, dtype=dtype)
        t = up.zero_apotheosis_tensor(x)
        s = up.zero_apotheosis_scalar(x)
        assert t.shape == x.shape
        assert s.shape == torch.Size([])
        assert torch.isfinite(t).all()
        assert torch.isfinite(s).all()

if __name__ == "__main__":
    pytest.main([__file__])
