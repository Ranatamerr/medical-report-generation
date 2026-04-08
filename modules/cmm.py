import torch
import torch.nn as nn
import torch.nn.functional as F


class CMM(nn.Module):
    """
    Cross-Modal Memory module.

    Enriches HAR anatomical features by having them query a learned
    memory bank that bridges visual and text semantic spaces.

    Input:  HAR region features  [batch, 8, d_vf]   (e.g. d_vf=2048)
    Output: enriched features    [batch, 8, d_vf]   (same shape)

    Steps:
      1. Project down: d_vf (2048) → d_model (512)  — enter text space
      2. Query memory bank with cross-attention
      3. Project back up: d_model (512) → d_vf (2048)
      4. Add residually to original HAR features
    """

    def __init__(self, d_vf, d_model, cmm_size=2048, num_heads=8, dropout=0.1):
        super(CMM, self).__init__()

        self.d_vf = d_vf
        self.d_model = d_model

        # Project HAR features into text semantic space
        self.down_proj = nn.Linear(d_vf, d_model)

        # Learned memory bank [cmm_size, d_model]
        self.memory = nn.Parameter(torch.randn(cmm_size, d_model))

        # Cross-attention: HAR features (as queries) attend to memory
        self.attn_q = nn.Linear(d_model, d_model)
        self.attn_k = nn.Linear(d_model, d_model)
        self.attn_v = nn.Linear(d_model, d_model)
        self.attn_out = nn.Linear(d_model, d_model)

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Project back to visual feature space
        self.up_proj = nn.Linear(d_model, d_vf)

    def forward(self, har_features):
        """
        har_features: [batch, num_regions, d_vf]
        returns:      [batch, num_regions, d_vf]
        """
        batch_size, num_regions, _ = har_features.shape

        # Step 1: project down to d_model
        x = self.down_proj(har_features)          # [batch, 8, d_model]

        # Step 2: cross-attention with memory bank
        memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, cmm_size, d_model]

        Q = self.attn_q(x)                        # [batch, 8, d_model]
        K = self.attn_k(memory)                   # [batch, cmm_size, d_model]
        V = self.attn_v(memory)                   # [batch, cmm_size, d_model]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_regions, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)               # [batch, heads, 8, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, num_regions, self.d_model)
        out = self.attn_out(out)

        # Residual + norm in text space
        x = self.norm(x + self.dropout(out))      # [batch, 8, d_model]

        # Step 3: project back to visual feature space
        x = self.up_proj(x)                       # [batch, 8, d_vf]

        # Step 4: residual connection with original HAR features
        return har_features + x
