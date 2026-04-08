import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from modules.cmm import CMM

print("Testing CMM module...")

batch_size = 2
num_regions = 8
d_vf = 2048
d_model = 512

# Simulate HAR output
fake_har_output = torch.randn(batch_size, num_regions, d_vf)
print(f"Input shape:  {fake_har_output.shape}")

# Create CMM module
cmm = CMM(d_vf=d_vf, d_model=d_model, cmm_size=2048, num_heads=8)

# Forward pass
output = cmm(fake_har_output)

print(f"Output shape: {output.shape}")

# Checks
assert output.shape == (batch_size, num_regions, d_vf), "SHAPE MISMATCH!"
assert not torch.equal(output, fake_har_output), "OUTPUT SHOULD DIFFER FROM INPUT!"

print("\nExpected:")
print(f"  Output shape: torch.Size([{batch_size}, {num_regions}, {d_vf}])")
print("\nAll tests passed!")
