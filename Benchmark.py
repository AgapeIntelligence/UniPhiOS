"""
UniPhiOS Benchmark: float32 vs float64 throughput
"""

import time
import torch
from uniphi_os.engine import GenesisGeometry

def run_benchmark(device: str = "cpu", dtype: torch.dtype = torch.float32, cycles: int = 50, batch: int = 4):
    device_obj = torch.device(device)
    model = GenesisGeometry(device=device, api_key=None, dtype=dtype).to(device_obj)
    model.train()

    # Input tensor
    x = torch.randn((batch, 512), dtype=dtype, device=device_obj)

    # Optimizer for a dummy training step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Warm-up
    for _ in range(5):
        bloom, identity_next, *_ = model(x)
        loss = torch.abs(bloom).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()

    # Benchmark loop
    start = time.time()
    for _ in range(cycles):
        bloom, identity_next, *_ = model(x)
        loss = torch.abs(bloom).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    samples_per_sec = (cycles * batch) / elapsed
    return {"device": device, "dtype": str(dtype), "cycles": cycles, "batch": batch, "elapsed_s": elapsed, "samples_per_s": samples_per_sec}

if __name__ == "__main__":
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for dev in devices:
        for dt in [torch.float32, torch.float64]:
            result = run_benchmark(dev, dt)
            print(f"{dev.upper()} {dt}: {result}")
