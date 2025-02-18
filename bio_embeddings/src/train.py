import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from embeddings.embeddings import CellLineEmbedding, DrugEmbedding
from model import DrugCellTransformer

### 학습시킬 데이터 가져오기 ###
class DrugResponseData(Dataset):
    def __init__(self, combined_cell_name, drug_smiles , labels, cell_embedding_method, drug_embedding_method):
        self.combined_cell_name = combined_cell_name
        self.drug_smiles = drug_smiles
        self.labels = labels
        self.cell_embedding = CellLineEmbedding(scbert_model_path="scbert_pretrained.pth")
        self.drug_embedding = DrugEmbedding()
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
            drug_features = self.drug_embedding.Fingerprint().morganFP([drug_smile])
        elif self.drug_embedding_method == "ecfp":
            drug_features = self.drug_embedding.Fingerprint().ECFP([drug_smile])
        elif self.drug_embedding_method == "gnn":
            drug_features = self.drug_embedding.GraphEmbedding().embed([drug_smile])
        elif self.drug_embedding_method == "selfies":
            drug_features = self.drug_embedding.fromSMILES().selfies([drug_smile])
        elif self.drug_embedding_method == "safe":
            drug_features = self.drug_embedding.fromSMILES().safe([drug_smile])

        return {"cell": cell_features, "drug": drug_features, "label": label}

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    combined_cell_name = df['combined_cell_line'].tolist()
    drug_smiles = df['drug_smiles'].tolist()
    labels = df['ln_ic50'].tolist()
    return combined_cell_name, drug_smiles, labels

### 모델 학습 ###
def train_model(combined_cell_name, drug_smiles, labels, cell_embedding_method, drug_embedding_method, num_epochs=100, batch_size=32, lr=1e-4, patience=10):
    dataset = DrugResponseData(combined_cell_name, drug_smiles, labels, cell_embedding_method, drug_embedding_method)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DrugCellTransformer(cell_embedding_dim=768, drug_embedding_dim=2048, hidden_dim=256, output_dim=1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            cell_features = batch['cell'].squeeze(1).float()
            drug_features = batch['drug'].squeeze(1).float()
            labels = batch['label'].float()

            outputs = model(drug_features, cell_features)
            loss = criterion(outputs.squeeze(1), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    return model

### GDSC2_cleaned.csv 가지고 학습시키기 ###
if __name__ == "__main__":
    file_path = "data/processed/GDSC2_cleaned.csv"
    combined_cell_name, drug_smiles, labels = load_dataset(file_path)

    # embedding methods 골라주기
    cell_embedding_method = "bioBERT"
    drug_embedding_method = "selfies"

    model = train_model(
        combined_cell_name,
        drug_smiles,
        labels,
        cell_embedding_method,
        drug_embedding_method,
        num_epochs=100,
        batch_size=32,
        lr=1e-4,
        patience=10
    )