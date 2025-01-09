import torch
import torch.nn as nn

class DrugResponsePredictor(nn.Module):
    def __init__(self, cell_line_embedding_dim, drug_embedding_dim, hidden_dim=128):
        """
        param cell_line_embedding_dim: Dimension of the cell line embedding input
        param drug_embedding_dim: Dimension of the drug embedding input
        param hidden_dim: Dimension of the hidden layers
        """
        super(DrugResponsePredictor, self).__init__()

        # fully connected layers for cell line and drug embeddings
        self.cell_line_fc = nn.Linear(cell_line_embedding_dim, hidden_dim)
        self.drug_fc = nn.Linear(drug_embedding_dim, hidden_dim)

        # transformer encoder to process combined embeddings
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # fully connected output layer
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, cell_line_embedding, drug_embedding):
        """
        param cell_line_embedding: Tensor containing cell line embeddings (batch_size, cell_line_embedding_dim)
        param drug_embedding: Tensor containing drug embeddings (batch_size, drug_embedding_dim)
        return: Predicted ln_ic50 value (batch_size, 1)
        """
        # pass through fully connected layers
        cell_line_out = torch.relu(self.cell_line_fc(cell_line_embedding))
        drug_out = torch.relu(self.drug_fc(drug_embedding))

        # combine embeddings
        combined = cell_line_out + drug_out
        combined = combined.unsqueeze(0)        # (1, batch_size, hidden_dim)

        # transformer encoder
        encoded_output = self.transformer_encoder(combined)
        encoded_output = encoded_output.squeeze(0)    # (batch_size, hidden_dim)

        output = self.output_layer(encoded_output)

        return output

### Example usage ###
if __name__ == "__main__":
    # Define dimensions based on embeddings.py output
    cell_line_embedding_dim = 200  # Example dimension for Word2Vec
    drug_embedding_dim = 768     # Example dimension for scBERT/ChemBERTa

    # Initialize model
    model = DrugResponsePredictor(cell_line_embedding_dim, drug_embedding_dim)

    # Example input data
    batch_size = 4
    cell_line_embedding = torch.randn(batch_size, cell_line_embedding_dim)  # Example random input
    drug_embedding = torch.randn(batch_size, drug_embedding_dim)           # Example random input

    # Forward pass
    predictions = model(cell_line_embedding, drug_embedding)
    print("Predicted ln_ic50:", predictions)
