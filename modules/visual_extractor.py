import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()

        # Original R2Gen ResNet-101 visual extractor
        resnet = models.resnet101(pretrained=True)

        # Remove final FC and avgpool layers — keep feature extractor only
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Global average pooling for fc_feats
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Project to d_vf (2048 → 2048, identity by default)
        self.projection = nn.Linear(2048, args.d_vf)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, images):
        # images: [batch, 3, 224, 224]

        # Extract feature map: [batch, 2048, 7, 7]
        feat_map = self.resnet(images)

        # Global average pool: [batch, 2048]
        avg_feats = self.avg_pool(feat_map).squeeze(-1).squeeze(-1)
        fc_feats = self.dropout(self.projection(avg_feats))

        # Flatten spatial: [batch, 49, 2048]
        batch_size, C, H, W = feat_map.shape
        att_feats = feat_map.permute(0, 2, 3, 1).reshape(batch_size, H * W, C)
        att_feats = self.dropout(self.projection(att_feats))

        return att_feats, fc_feats
