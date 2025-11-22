"""
UniPhiOS Unit Tests: Shapes, Dtypes, and Numeric Invariants
"""

import torch
from uniphi_os.engine import GenesisGeometry

def test_forward_shapes_and_dtype(dtype):
    device = "cpu"
    model = GenesisGeometry(device=device, api_key=None, dtype=dtype)
    x = torch.randn((1, 512), dtype=dtype)

    bloom, identity_next, crown, triad, spiral, echo = model(x)

    # Shape assertions
    assert bloom.shape == torch.Size([1, 1])
    assert identity_next.shape == torch.Size([1, 512])
    assert crown.shape == torch.Size([1, 1])
    assert triad.shape == torch.Size([1, 3])
    assert spiral.shape == torch.Size([1, 108])

    # Dtype assertions
    assert bloom.dtype == dtype
    assert identity_next.dtype == dtype

    # Numeric sanity
    assert torch.isfinite(bloom).all()
    assert torch.isfinite(identity_next).all()

    print(f"Test {dtype}: PASS - Shapes & Finites Sovereign")

if __name__ == "__main__":
    test_forward_shapes_and_dtype(torch.float32)
    test_forward_shapes_and_dtype(torch.float64)
