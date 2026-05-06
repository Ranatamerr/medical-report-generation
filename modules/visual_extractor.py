import torch
import torch.nn as nn
from open_clip import create_model_from_pretrained


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()

        # Load BiomedCLIP vision encoder (medically pretrained on 15M biomedical image-text pairs)
        # Paper: https://arxiv.org/abs/2303.00915
        # First run downloads ~1GB and caches in ~/.cache/huggingface/
        model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.vit = model.visual.trunk  # timm ViT backbone, same patch geometry as google/vit-base-patch16-224

        # BiomedCLIP ViT outputs 768-dim features; decoder expects 2048-dim
        # Separate projections: patches are local, CLS is global summary
        self.project_patch = nn.Linear(768, args.d_vf)
        self.project_cls   = nn.Linear(768, args.d_vf)

    def forward(self, images):
        # images shape: [batch, 3, 224, 224]

        # timm ViT forward_features returns [batch, 197, 768]
        # 197 = 1 CLS token + 196 patch tokens (14x14 grid)
        all_tokens = self.vit.forward_features(images)

        cls_token    = all_tokens[:, 0, :]   # [batch, 768]
        patch_tokens = all_tokens[:, 1:, :]  # [batch, 196, 768]

        # Project from 768 → 2048 using separate weights
        patch_feats = self.project_patch(patch_tokens)  # [batch, 196, 2048]
        avg_feats   = self.project_cls(cls_token)        # [batch, 2048]

        return patch_feats, avg_feats
