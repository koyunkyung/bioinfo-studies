import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F
from message_passing import *
from readout import *

class MPNNModel(nn.Module):
    def __init__(
        self,
        atom_dim,
        bond_dim,
        batch_size=32,
        message_units=64,
        message_steps=4,
        num_attention_heads=8,
        dense_units=512
    ):
        super(MPNNModel, self).__init__()
        
        self.message_passing = MessagePassing(
            units=message_units,
            steps=message_steps
        )

        self.transformer_readout = TransformerEncoderReadout(
            num_heads=num_attention_heads,
            embed_dim=message_units,
            dense_dim=dense_units,
            batch_size=batch_size
        )

        self.dense1 = nn.Linear(dense_units, dense_units)
        self.dense2 = nn.Linear(dense_units, 1)

        self.message_passing.build([
            (None, atom_dim),
            (None, bond_dim),
            (None, 2)
        ])

    def forward(self, inputs):
        atom_features, bond_features, pair_indices, molecule_indicator = inputs
        x = self.message_passing([atom_features, bond_features, pair_indices])
        x = self.transformer_readout([x, molecule_indicator])
        x = F.relu(self.dense1(x))
        x = torch.sigmoid(self.dense2(x))
        return x
    
def model_set(x_train, learning_rate=5e-4):
    model = MPNNModel(atom_dim=x_train[0][0][0].shape[0], bond_dim=x_train[1][0][0].shape[0])
    optimizer = Adam(model.parameters(), lr = learning_rate)
    criterion = nn.BCELoss
    return model, optimizer, criterion
