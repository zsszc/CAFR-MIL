import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextAwareFeatureRefiner(nn.Module):
    """
    [Innovation Module] Context-Aware Feature Refinement (CAFR)
    Purpose:
    1. Aggregate bag-level global context.
    2. Dynamically generate channel attention weights to suppress noisy channels
       (e.g., staining background) and enhance pathological semantic channels.
    3. Keep input and output dimensions identical for plug-and-play usage.
    """
    def __init__(self, input_dim=256, reduction=16):
        super(ContextAwareFeatureRefiner, self).__init__()
        # Bottleneck dimension; prevent excessive compression and keep at least 16
        mid_dim = max(input_dim // reduction, 16)

        self.channel_attention = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, input_dim),
            nn.Sigmoid()
        )
        # LayerNorm stabilizes feature distribution, crucial for Transformer backbones
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x shape: [Batch, N_instances, Dim]

        # 1. Global Context Modeling
        # Average over N instances to obtain bag-level descriptor
        context = torch.mean(x, dim=1, keepdim=True)  # [B, 1, D]

        # 2. Dynamic Weight Generation
        # Learn channel-wise importance
        weights = self.channel_attention(context)  # [B, 1, D]

        # 3. Feature Recalibration
        # Apply weights to all instances via broadcasting
        x_refined = x * weights

        # 4. Residual Connection
        # Preserve original features and perform "enhancement" only
        out = x + x_refined

        return self.ln(out)


class OrthogonalLoss(nn.Module):
    """
    [Innovation Loss] Orthogonal Loss (feature decorrelation regularization)
    Purpose:
    Enforce feature channels to be mutually orthogonal (uncorrelated),
    reducing redundancy and forcing the network to learn diverse pathological cues.
    """
    def __init__(self):
        super(OrthogonalLoss, self).__init__()

    def forward(self, features):
        # features: [B, N, D]

        # If batch size > 1, concatenate all bag instances for computation
        if features.dim() == 3:
            features = features.view(-1, features.size(2))  # [B*N, D]

        # 1. Column-wise normalization
        # Remove magnitude effect, keep only direction (correlation)
        norm = torch.norm(features, p=2, dim=0, keepdim=True)  # [1, D]
        features_norm = features / (norm + 1e-8)

        # 2. Compute Gram matrix (correlation matrix)
        # Shape [D, D], entry (i, j) is cosine similarity between channel i and j
        gram_matrix = torch.mm(features_norm.t(), features_norm)

        # 3. Build target identity matrix
        # Only diagonal should be 1, off-diagonal 0
        eye = torch.eye(gram_matrix.shape[0], device=gram_matrix.device)

        # 4. Frobenius norm as loss
        D = gram_matrix.shape[0]
        loss = torch.norm(gram_matrix - eye, p='fro') / (D * D)

        return loss
```