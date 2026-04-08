import torch
import torch.nn as nn
import torch.nn.functional as F


# Anatomical regions for chest X-rays
ANATOMICAL_REGIONS = [
    'left_lung',
    'right_lung', 
    'heart',
    'mediastinum',
    'left_pleural',
    'right_pleural',
    'bones',
    'diaphragm'
]

NUM_REGIONS = len(ANATOMICAL_REGIONS)  # 8


class HARLayer(nn.Module):
    """
    One layer of Hierarchical Anatomical Reasoning.
    Uses cross-attention to extract features for each anatomical region.
    
    Input:  patch features from ViT [batch, 196, d_vf]
    Output: region features          [batch, num_regions, d_vf]
    """

    def __init__(self, d_vf, num_heads=8, dropout=0.1):
        super(HARLayer, self).__init__()

        self.d_vf = d_vf
        self.num_heads = num_heads
        self.head_dim = d_vf // num_heads

        # Cross-attention: regions attend to ViT patches
        self.q_proj = nn.Linear(d_vf, d_vf)  # queries = region embeddings
        self.k_proj = nn.Linear(d_vf, d_vf)  # keys   = ViT patches
        self.v_proj = nn.Linear(d_vf, d_vf)  # values = ViT patches
        self.out_proj = nn.Linear(d_vf, d_vf)

        self.norm1 = nn.LayerNorm(d_vf)
        self.norm2 = nn.LayerNorm(d_vf)

        self.ffn = nn.Sequential(
            nn.Linear(d_vf, d_vf * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_vf * 2, d_vf)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, region_queries, patch_features):
        """
        region_queries:  [batch, num_regions, d_vf]
        patch_features:  [batch, num_patches, d_vf]
        """
        batch_size = patch_features.size(0)

        # Cross-attention
        Q = self.q_proj(region_queries)
        K = self.k_proj(patch_features)
        V = self.v_proj(patch_features)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Attended features
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_vf)
        out = self.out_proj(out)

        # Residual + norm
        region_queries = self.norm1(region_queries + self.dropout(out))

        # FFN + residual
        region_queries = self.norm2(region_queries + self.dropout(self.ffn(region_queries)))

        return region_queries


class ACALoss(nn.Module):
    """
    Anatomical Consistency Alignment loss.
    Uses contrastive learning to:
    - Pull same regions together across patients (heart_A ~ heart_B)
    - Push different regions apart (heart != lung)
    """

    def __init__(self, temperature=0.07):
        super(ACALoss, self).__init__()
        self.temperature = temperature

    def forward(self, region_features):
        """
        region_features: [batch, num_regions, d_vf]
        
        For each region type, features from different patients 
        should be similar. Features from different regions 
        should be different.
        """
        batch_size, num_regions, d_vf = region_features.shape

        # Normalize features
        region_features = F.normalize(region_features, dim=-1)

        loss = 0.0
        count = 0

        for r in range(num_regions):
            # Features for region r across all patients: [batch, d_vf]
            region_r = region_features[:, r, :]

            # Similarity matrix: [batch, batch]
            sim_matrix = torch.matmul(region_r, region_r.T) / self.temperature

            # Positive pairs: same region, different patients (diagonal excluded)
            # Negative pairs: different regions, same or different patients
            labels = torch.arange(batch_size).to(region_features.device)

            loss += F.cross_entropy(sim_matrix, labels)
            count += 1

        return loss / count


class HAR(nn.Module):
    """
    Hierarchical Anatomical Reasoning module.
    
    Takes ViT patch features and produces structured 
    anatomical region features using 4 stacked layers.
    
    Also computes ACA loss for training.
    """

    def __init__(self, d_vf, num_layers=4, num_heads=8, dropout=0.1):
        super(HAR, self).__init__()

        self.num_regions = NUM_REGIONS
        self.d_vf = d_vf

        # Learnable region query embeddings (one per anatomical region)
        self.region_embeddings = nn.Embedding(NUM_REGIONS, d_vf)

        # 4 stacked HAR layers
        self.layers = nn.ModuleList([
            HARLayer(d_vf, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # ACA loss
        self.aca_loss = ACALoss()

        # Final projection
        self.output_norm = nn.LayerNorm(d_vf)

    def forward(self, patch_features, compute_aca=True):
        """
        patch_features: [batch, num_patches, d_vf]  ← comes from ViT
        
        Returns:
            region_features: [batch, num_regions, d_vf]  ← goes to CMN/decoder
            aca_loss:        scalar (0 during inference)
        """
        batch_size = patch_features.size(0)

        # Initialize region queries from learnable embeddings
        region_ids = torch.arange(self.num_regions).to(patch_features.device)
        region_queries = self.region_embeddings(region_ids)  # [num_regions, d_vf]
        region_queries = region_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_regions, d_vf]

        # Pass through 4 HAR layers
        for layer in self.layers:
            region_queries = layer(region_queries, patch_features)

        # Final normalization
        region_features = self.output_norm(region_queries)

        # Compute ACA loss if training
        aca_loss = torch.tensor(0.0).to(patch_features.device)
        if compute_aca and self.training and batch_size > 1:
            aca_loss = self.aca_loss(region_features)

        return region_features, aca_loss