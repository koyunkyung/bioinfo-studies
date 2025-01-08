import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

### Unified Transformer Model ###
class UnifiedTransformer(nn.Module):
    def __init__(self, embedding_dim_cell, embedding_dim_drug, hidden_dim, output_dim, 
                 drug_embedding_type, scbert_model_name="havens2/scBERT_SER", rdkit_feature_dim=4):
        """
        Unified Transformer Model to support different embedding types
        - "label": LabelEncoder-based embeddings
        - "scbert": scBERT-based embeddigns
        - "rdkit": RDKit molecular feature embeddings
        """
        super(UnifiedTransformer, self).__init__()

        # Cell line embedding (precomputed)
        self.cell_line_fc = nn.Linear(embedding_dim_cell, hidden_dim)

        # Drug name embedding (flexible: scBERT or RDKit)
        if drug_embedding_type == "scbert":
            self.drug_embedding = SCBERTEmbedding(scbert_model_name)
            self.drug_emb_dim = self.drug_embedding.hidden_dim
        elif drug_embedding_type == "rdkit":
            self.drug_embedding = RDKitEmbedding(rdkit_feature_dim, embedding_dim_drug)
            self.drug_emb_dim = embedding_dim_drug
        else:
            raise ValueError(f"Unsupported drug embedding type: {drug_embedding_type}")
        
        # Transformer Encoder
        self.transformer = nn.Transformer(
            d_model=hidden_dim + self.drug_emb_dim,
            nhead=8,
            num_encoder_layers=4,
            dim_feedforward=hidden_dim * 2,
            batch_first=True
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim + self.drug_emb_dim, output_dim)

    def forward(self, cell_line_inputs, drug_inputs):
        cell_line_emb = self.cell_line_fc(cell_line_inputs)                 # (batch_size, hidden_dim)
        drug_emb = self.drug_embedding(drug_inputs)                         # shape depends on the embedding type
        combined = torch.cat((cell_line_emb, drug_emb), dim=1).unsqueeze(1) # shape: (batch_size, 1, combined_dim)
        transformer_out = self.transformer(combined, combined)

        output = self.fc(transformer_out[:, 0, :])
        return output




### Drug Name Embedding: SCBERT ###
class SCBERTEmbedding(nn.Module):
    def __init__(self, model_name="havens2/scBERT_SER"):
        super(SCBERTEmbedding, self).__init__()
        self.scbert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_dim = self.scbert.config.hidden_size

    def forward(self, smiles_list):
        assert isinstance(smiles_list, list) and all(isinstance(s, str) for s in smiles_list), \
            "smiles_list must be a list of strings (SMILES)."
        tokenized_inputs = self.tokenizer(
            smiles_list, padding=True, truncation=True, return_tensors="pt"
        )

        tokenized_inputs = {key: value.to(next(self.scbert.parameters()).device)
                            for key, value in tokenized_inputs.items()}

        outputs = self.scbert(**tokenized_inputs)
        return outputs.last_hidden_state[:, 0, :]
    

### Drug Name Embedding: RDKit ###
class RDKitEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(RDKitEmbedding, self).__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)

    def forward(self, molecular_features):
        return self.fc(molecular_features)