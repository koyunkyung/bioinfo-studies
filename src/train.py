import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from preprocess.embeddings import CellLineEmbedding, DrugEmbedding
from model import DrugResponsePredictor
from sklearn.preprocessing import OneHotEncoder
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

import os
from transformers import logging

# Suppress TOKENIZERS_PARALLELISM warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress Hugging Face warnings
logging.set_verbosity_error()

### Dataset class with modular drug embedding ###
class DrugResponseDataset(Dataset):
    def __init__(self, cell_lines, diseases, drugs, drug_names, targets, ln_ic50, drug_embedding_method):
        """
        param cell_lines: List of cell line names
        param diseases: List of disease names corresponding to the cell lines
        param drugs: List of drug information (e.g., SMILES strings or names)
        param ln_ic50: List of ln_ic50 target values
        param drug_embedding_method: Drug embedding method ('one_hot', 'scbert', 'chemberta')
        """
        self.cell_lines = cell_lines
        self.diseases = diseases
        self.drugs = drugs
        self.drug_names = drug_names
        self.targets = targets
        self.ln_ic50 = ln_ic50
        self.drug_embedding_method = drug_embedding_method

        combined_drug_info = [f"{name}_{target}" for name, target in zip(drug_names, targets)]
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.one_hot_encoder.fit(np.array(combined_drug_info).reshape(-1, 1))

        self.one_hot_vector_size = len(self.one_hot_encoder.categories_[0])

        # initialize embeddings
        self.cell_line_embedding = CellLineEmbedding(vector_size=100, window=5, min_count=1, epochs=100)
        self.drug_embedding = DrugEmbedding()

        # train Word2Vec model for cell line embeddings
        cell_line_disease_data = [[cl, d] for cl, d in zip(cell_lines, diseases)]
        self.word2vec_model = self.cell_line_embedding.train_word2vec(cell_line_disease_data)

    def __len__(self):
        return len(self.cell_lines)

    def __getitem__(self, idx):
        # Cell line embedding
        cell_line_vector = self.cell_line_embedding.get_embedding(self.word2vec_model, self.cell_lines[idx])
        disease_vector = self.cell_line_embedding.get_embedding(self.word2vec_model, self.diseases[idx])
        cell_line_embedding = torch.tensor(cell_line_vector + disease_vector, dtype=torch.float32)  # Combine embeddings

        # Drug embedding based on the selected method
        if self.drug_embedding_method == "one_hot":
            combined_drug_info = f"{self.drug_names[idx]}_{self.targets[idx]}"
            drug_vector = self.one_hot_encoder.transform([[combined_drug_info]])[0]  # Use fitted encoder
        elif self.drug_embedding_method == "scbert":
            drug_vector = self.drug_embedding.scbert_embedding([self.drugs[idx]])[0].numpy()
        elif self.drug_embedding_method == "chemberta":
            drug_vector = self.drug_embedding.chemberta_embedding([self.drugs[idx]])[0].numpy()
        else:
            raise ValueError(f"Unsupported drug embedding method: {self.drug_embedding_method}")

        drug_embedding = torch.tensor(drug_vector, dtype=torch.float32)

        # Target ln_ic50 value
        ln_ic50_value = torch.tensor(self.ln_ic50[idx], dtype=torch.float32)

        return cell_line_embedding, drug_embedding, ln_ic50_value

def train_model(model, dataloader, criterion, optimizer, num_epochs=100, patience=10, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.to(device)
    best_loss = float('inf')  # Initialize the best loss to a very high value
    epochs_without_improvement = 0  # Track the number of epochs with no improvement

    # Optimizer, scheduler, and scaler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for cell_line_emb, drug_emb, ln_ic50 in dataloader:
            cell_line_emb = cell_line_emb.to(device)
            drug_emb = drug_emb.to(device)
            ln_ic50 = ln_ic50.to(device)

            # Forward pass
            optimizer.zero_grad()
            predictions = model(cell_line_emb, drug_emb)
            loss = criterion(predictions.squeeze(), ln_ic50)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0  # Reset counter if loss improves
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

### Main Script ###
if __name__ == "__main__":
    data = pd.read_csv("data/processed/GDSC2_cleaned.csv")
    # Extract required columns
    cell_lines = data["cell_line_name"].tolist()
    diseases = data["disease"].tolist()
    drugs = data["smiles"].tolist()
    drug_names = data["drug_name"].tolist()
    targets = data["putative_target"].tolist()
    ln_ic50 = data["ln_ic50"].tolist()

    # Drug embedding methods to compare
    embedding_methods = ["one_hot", "scbert", "chemberta"]

    # Iterate over each embedding method
    for method in embedding_methods:
        print(f"\n=== Training with {method} drug embeddings ===\n")

        # Create dataset and dataloader
        dataset = DrugResponseDataset(cell_lines, diseases, drugs, drug_names, targets, ln_ic50, drug_embedding_method=method)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Dynamically compute one-hot vector size
        if method == "one_hot":
            drug_embedding_dim = dataset.one_hot_vector_size
        elif method == "scbert":
            drug_embedding_dim = 768
        elif method == "chemberta":
            drug_embedding_dim = 768
        else:
            raise ValueError(f"Unknown embedding method: {method}")

        # Initialize model
        cell_line_embedding_dim = 100  # Word2Vec embedding size
        model = DrugResponsePredictor(cell_line_embedding_dim, drug_embedding_dim)

        # Define loss function and optimizer
        criterion = nn.MSELoss()  # Mean Squared Error for regression
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train model
        train_model(model, dataloader, criterion, optimizer, num_epochs=100, device="cpu")
