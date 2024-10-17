import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn
from torch_scatter import scatter_mean, scatter_std

def batch_normalize(x, batch):
    mean = scatter_mean(x, batch, dim=0)[batch]
    std = scatter_std(x, batch, dim=0)[batch]
    return (x - mean) / (std + 1e-8)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_freq=10.0, num_freq_bands=64):
        super().__init__()
        self.d_model = d_model
        freq_bands = torch.linspace(1., max_freq, num_freq_bands)
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, xyz):
        freq_bands = self.freq_bands.view(1, 1, -1) 
        x_freq = xyz.unsqueeze(-1) * freq_bands
        sin_encoded = torch.sin(x_freq)
        cos_encoded = torch.cos(x_freq)
        encoded = torch.stack([sin_encoded, cos_encoded], dim=-1)
        encoded = encoded.view(*xyz.shape[:-1], -1)  
        if encoded.shape[-1] > self.d_model:
            encoded = encoded[..., :self.d_model]
        return encoded

class AdaptiveBlending(nn.Module):
    def __init__(self, feature_dim, pos_dim, attention_dim=64):
        super().__init__()
        self.feature_attention = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1)
        )
        self.pos_attention = nn.Sequential(
            nn.Linear(pos_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, features, pos):
        feature_weight = self.feature_attention(features)
        pos_weight = self.pos_attention(pos)
        combined_weights = torch.cat([feature_weight, pos_weight], dim=-1)
        soft_weights = torch.softmax(combined_weights, dim=-1)
        feature_weight, pos_weight = soft_weights.split(1, dim=-1)
        return feature_weight, pos_weight

class ATSearchKNN(nn.Module):
    def __init__(self, k, attention_dim=32):
        super().__init__()
        self.k = k
        self.attention_dim = attention_dim
        self.encoding = PositionalEncoding(attention_dim)
        
    def forward(self, x, pos, batch, focal_points=None):
        #reflectance = pos[:, 3].unsqueeze(1)
        #x = torch.cat([x, reflectance], dim=1)
                
        encoding = self.encoding(pos[:, :3]).to(pos.device)
        
        features_encoded = torch.cat([x, encoding], dim=-1)
        pos_encoded = torch.cat([pos[:, :3], encoding], dim=-1)
        
        adaptive_blending = AdaptiveBlending(features_encoded.shape[1], pos_encoded.shape[1], self.attention_dim).to(x.device)
        feature_weight, pos_weight = adaptive_blending(features_encoded, pos_encoded)
        
        weighted_features = features_encoded * feature_weight
        weighted_pos = pos_encoded * pos_weight
        
        combined = torch.cat([weighted_pos, weighted_features], dim=-1)
        combined_normalized = batch_normalize(combined, batch)
        
        if focal_points is None:
            focal_points = torch.arange(pos.size(0), device=pos.device)
        batch_focal = batch[focal_points]
        
        row, col = knn(combined_normalized, combined_normalized[focal_points], 
                       k=self.k, batch_x=batch, batch_y=batch_focal)
        return row, col
