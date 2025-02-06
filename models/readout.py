import torch
import torch.nn as nn
import torch.nn.functional as F

## 분자별 원자 특성 분할 및 패딩 
class PartitionPadding(nn.Module):
    def __init__(self, batch_size):
        super(PartitionPadding, self).__init__()
        self.batch_size = batch_size

    def forward(self, inputs):
        atom_features, molecule_indicator = inputs

        # 분자별 원자 특성 분할
        features_partitioned = []
        for i in range(self.batch_size):
            mask = (molecule_indicator == i)
            features_partitioned.append(atom_features[mask])

        # padding
        num_atoms = [feat.size(0) for feat in features_partitioned]
        max_num_atoms = max(num_atoms)

        features_padded = []
        for feat, n_atoms in zip(features_partitioned, num_atoms):
            padding_size = max_num_atoms - n_atoms
            if padding_size > 0:
                padding = (0, 0, 0, padding_size)
                feat_padded = F.pad(feat, padding)
            else:
                feat_padded = feat
            features_padded.append(feat_padded)

        # stacking
        features_stacked = torch.stack(features_padded)
        mask = torch.sum(torch.sum(features_stacked, dim=2), dim=1) != 0
        features_stacked = features_stacked[mask]
        return features_stacked
    

## 트랜스포머모델 이용해서 그래프 수준의 특성 추출하기 ##
class TransformerEncoderReadout(nn.Module):
    def __init__(self, num_heads=8, embed_dim=64, dense_dim=512, batch_size=32):
        super(TransformerEncoderReadout, self).__init__()
        self.partition_padding = PartitionPadding(batch_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.dense_proj = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.average_pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, inputs):
        x = self.partition_padding(inputs)
        padding_mask = torch.any(x != 0, dim=-1)
        attention_mask = padding_mask.unsqueeze(1).unsqueeze(2)

        attention_output, _ = self.attention(
            x, x, x,
            key_padding_mask = ~padding_mask
        )
        proj_input = self.layernorm1(x + attention_output)
        proj_output = self.layernorm2(proj_input + self.dense_proj(proj_input))
        pooled = self.average_pooling(proj_output.transpose(1, 2)).squeeze(-1)
        return pooled
    