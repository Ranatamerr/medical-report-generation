import os
import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()

        # Load pretrained Xception via timm
        import timm
        self.xception = timm.create_model('xception', pretrained=True, num_classes=0, global_pool='')
        # Xception outputs [batch, 2048, 10, 10] for 299x299 input
        # For 224x224 input it outputs [batch, 2048, 7, 7] = 49 spatial locations

        # Project from 2048 → d_vf (2048 by default, so this is identity-like)
        xception_dim = 2048
        self.project = nn.Linear(xception_dim, args.d_vf)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        # images shape: [batch, 3, 224, 224]

        # Extract features — shape: [batch, 2048, H, W]
        feat_map = self.xception(images)

        batch_size, C, H, W = feat_map.shape

        # Flatten spatial dims → sequence of patch-like features
        # [batch, 2048, H, W] → [batch, H*W, 2048]
        patch_feats = feat_map.permute(0, 2, 3, 1).reshape(batch_size, H * W, C)

        # Project to d_vf
        patch_feats = self.project(patch_feats)  # [batch, H*W, d_vf]

        # Global average pooling for fc_feats (summary vector)
        avg = self.avg_pool(feat_map).squeeze(-1).squeeze(-1)  # [batch, 2048]
        avg_feats = self.project(avg)  # [batch, d_vf]

        return patch_feats, avg_feats
