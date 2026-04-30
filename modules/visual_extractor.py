import os
import torch
import torch.nn as nn
from transformers import ViTModel


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()

        # Load pretrained ViT
        # This downloads ~330MB the first time, then caches it
        vit_path = os.path.join(os.path.dirname(__file__), '..', 'vit_model')
        self.vit = ViTModel.from_pretrained(vit_path)

        # ViT outputs 768-dim features; decoder expects 2048-dim
        # Separate projections: patches are local, CLS is global summary
        self.project_patch = nn.Linear(768, args.d_vf)
        self.project_cls   = nn.Linear(768, args.d_vf)

    def forward(self, images):
        # images shape: [batch, 3, 224, 224]

        # Run ViT — outputs one vector per patch
        outputs = self.vit(pixel_values=images)

        # last_hidden_state shape: [batch, 197, 768]
        # 197 = 1 CLS token + 196 patch tokens
        all_tokens = outputs.last_hidden_state

        # Split CLS token (global summary) from patch tokens (local regions)
        cls_token    = all_tokens[:, 0, :]      # [batch, 768]
        patch_tokens = all_tokens[:, 1:, :]     # [batch, 196, 768]

        # Project from 768 → 2048 using separate weights
        patch_feats = self.project_patch(patch_tokens)  # [batch, 196, 2048]
        avg_feats   = self.project_cls(cls_token)        # [batch, 2048]

        # Return same variable names the decoder expects
        return patch_feats, avg_feats