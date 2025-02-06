import torch
import torch.nn as nn

## edge 통해 이웃 노드들의 정보 처리 네트워크 ##
class EdgeNetwork(nn.Module):
    def __init__(self):
        super(EdgeNetwork, self).__init__()
        self.atom_dim = None
        self.bond_dim = None
        self.linear = None

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]

        # 선형 층 이용해서 결합 특성 변환해야 함
        self.linear = nn.Linear(
            in_features = self.bond_dim,
            out_features = self.atom_dim * self.atom_dim
        )

    def forward(self, inputs):
        atom_features, bond_features, pair_indices = inputs

        bond_features = self.linear(bond_features)
        bond_features = bond_features.view(-1, self.atom_dim, self.atom_dim)

        atom_features_neighbors = torch.index_select(
            atom_features, 0, pair_indices[:, 1]
        )   # (num_bonds, atom_dim)
        atom_features_neighbors = atom_features_neighbors.squeeze(-1)

        # 이웃 특성과 변환된 결합 특성 결합
        transformed_features = torch.matmul(bond_features, atom_features_neighbors)
        transformed_features = transformed_features.squeeze(-1)

        aggregated_features = torch.zeros_like(atom_features)
        aggregated_features.index_add_(
            0, pair_indices[:, 0], transformed_features
        )
        return aggregated_features


## 메시지 전달 통해 원자 특성 업데이트 ##
class MessagePassing(nn.Module):
    def __init__(self, units, steps=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps
    
    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.pad_length = max(0, self.units - self.atom_dim)

        self.message_step = EdgeNetwork()
        self.message_step.build(input_shape)

        self.update_step = nn.GRUCell(
            input_size = self.atom_dim + self.pad_length,
            hidden_size = self.atom_dim + self.pad_length
        )

    def forward(self, inputs):
        atom_features, bond_features, pair_indices = inputs

        if self.pad_length > 0:
            atom_features_updated = nn.functional.pad(
                atom_features, (0, self.pad_length)
            )
        else:
            atom_features_updated = atom_features

        for _ in range(self.steps):
            atom_features_aggregated = self.message_step(
                [atom_features_updated, bond_features, pair_indices]
            )
            atom_features_updated = self.update_step(
                atom_features_aggregated,
                atom_features_updated
            )
        return atom_features_updated