import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class DrugResponseTransformer(nn.Module):
    def __init__(self, num_cell_lines, num_drugs, embed_dim, hidden_dim, output_dim):
        super(DrugResponseTransformer, self).__init__()

        # Embedding layers
        self.cell_line_embedding = nn.Embedding(num_cell_lines, embed_dim)
        self.drug_embedding = nn.Embedding(num_drugs, embed_dim)

         # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True),
            num_layers=2
         )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, cell_line, drug):

        # Embedding lookup
        cell_line_embed = self.cell_line_embedding(cell_line)
        drug_embed = self.drug_embedding(drug)

        # Combine embeddings into a sequence
        combined = torch.stack((cell_line_embed, drug_embed), dim=1)

        # Transformer encoder processing
        transformer_output = self.transformer(combined)     # shape: [batch_size, 2, embed_dim]
        pooled_output = transformer_output.mean(dim=1)      # shape: [batch_size, embed_dim]

        # Fully connected layers
        output = self.fc(pooled_output)
        return output
    