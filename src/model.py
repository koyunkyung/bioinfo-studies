import torch
import torch.nn as nn

class DrugCellTransformer(nn.Module):
    def __init__(self, cell_embedding_dim, drug_embedding_dim, hidden_dim, output_dim):
        super(DrugCellTransformer, self).__init__()

        # Input projection layers
        self.cell_projection = nn.Linear(cell_embedding_dim, hidden_dim)
        self.drug_projection = nn.Linear(drug_embedding_dim, hidden_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, drug_features, cell_features):
        # Project embeddings to hidden_dim
        drug_encoded = self.relu(self.drug_projection(drug_features))
        cell_encoded = self.relu(self.cell_projection(cell_features))

        # Concatenate embeddings
        combined = torch.cat((drug_encoded, cell_encoded), dim=1)

        # Fully connected layers
        x = self.relu(self.fc1(combined))
        output = self.fc2(x)

        return output