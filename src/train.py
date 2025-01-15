import torch
from torch.utils.data import Dataset, DataLoader
from embeddings import CellLineEmbedding, DrugEmbedding
from model import DrugCellTransformer

class DrugResponseData(Dataset):
    def __init__(self, combined_cell_name, combined_drug, labels, cell_embedding_method, drug_embedding_method):
        self.combined_cell_name = combined_cell_name
        self.combined_drug = combined_drug
        self.labels = labels

        # Embedding classes
        self.cell_embedding = CellLineEmbedding()
        self.drug_embedding = DrugEmbedding()

        # Embedding methods
        self.cell_embedding_method = cell_embedding_method
        self.drug_embedding_method = drug_embedding_method

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cell_name = self.combined_cell_name[idx]
        drug_smile = self.drug_smiles[idx]
        label = self.labels[idx]

        # Cell Line Embedding
        if self.cell_embedding_method == "scBERT":
            cell_features = self.cell_embedding.scBERT([cell_name])
        elif self.cell_embedding_method == "bioBERT":
            cell_features = self.cell_embedding.bioBERT([cell_name])

        # Drug Embedding
        if self.drug_embedding_method == "morganFP":
            drug_features = self.drug_embedding.morganFP([drug_smile])
        elif self.drug_embedding_method == "ecfp":
            drug_features = self.drug_embedding.ecfp([drug_smile])

        return {"cell": cell_features, "drug": drug_features, "label": label}

# Dataset and DataLoader
dataset = DrugResponseData(combined_cell_name, drug_smiles, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, Loss, Optimizer
model = DrugCellTransformer(drug_vocab_size=2048, cell_feature_size=768, hidden_dim=256, output_dim=1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
for epoch in range(10):
    model.train()
    total_loss = 0

    for batch in dataloader:
        cell_features = batch['cell'].squeeze(1)
        drug_features = batch['drug'].squeeze(1)
        labels = batch['label']

        outputs = model(drug_features, cell_features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
