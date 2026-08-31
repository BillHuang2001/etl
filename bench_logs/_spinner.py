#!/usr/bin/env python3
"""Keep a chosen GPU's clocks boosted during timing runs WITHOUT hogging it.

GPU boost is utilization-driven; the probe kernels are so short (0.06-0.2 ms
per call with ms-scale host-staging gaps) that the card idles at 210/405 MHz
clocks, which distorts per-call wall times (copy engines run off the memory
clock). A tiny periodic matmul (256x256 fp32 ~0.03 ms every 5 ms, <1% duty)
holds SM at ~1800 MHz / mem at 7600 MHz while sampling 0% utilization.

Usage: python _spinner.py <gpu-index> <seconds>   (torch selects cuda:<gpu>)
"""
import sys
import time

import torch

gpu = int(sys.argv[1])
seconds = float(sys.argv[2])
dev = torch.device(f"cuda:{gpu}")
torch.manual_seed(0)
a = torch.randn(256, 256, device=dev)
b = torch.randn(256, 256, device=dev)
t_end = time.time() + seconds
i = 0
while time.time() < t_end:
    torch.mm(a, b)
    time.sleep(0.005)
    i += 1
print(f"spinner: {i} iters on cuda:{gpu}", flush=True)
