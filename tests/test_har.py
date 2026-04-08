import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from modules.har import HAR

print("Testing HAR module...")

# Simulate ViT output
batch_size = 2
num_patches = 196
d_vf = 2048

fake_vit_output = torch.randn(batch_size, num_patches, d_vf)
print(f"Input shape:  {fake_vit_output.shape}")

# Create HAR module
har = HAR(d_vf=d_vf, num_layers=4, num_heads=8, dropout=0.1)
har.train()  # training mode to test ACA loss

# Forward pass
region_features, aca_loss = har(fake_vit_output, compute_aca=True)

print(f"Output shape: {region_features.shape}")
print(f"ACA loss:     {aca_loss.item():.4f}")

# Expected output
print("\nExpected:")
print(f"  Output shape: torch.Size([{batch_size}, 8, {d_vf}])")
print(f"  ACA loss:     a positive number")

# Check
assert region_features.shape == (batch_size, 8, d_vf), "SHAPE MISMATCH!"
assert aca_loss.item() > 0, "ACA LOSS SHOULD BE POSITIVE!"

print("\nAll tests passed!")