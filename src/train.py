import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import DrugResponseTransformer
import torch.optim as optim
import pandas as pd

# Dataset for DrugResponseTransformer
class DrugResponseDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.cell_lines = torch.tensor(data['cell_line_encoded'].values, dtype=torch.long)
        self.drugs = torch.tensor(data['drug_encoded'].values, dtype=torch.long)
        self.targets = torch.tensor(data['ln_ic50'].values, dtype=torch.float)

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'cell_line': self.cell_lines[idx],
            'drug': self.drugs[idx],
            'target': self.targets[idx]
        }

# Dataset for DrugResponseChem
class DrugResponseSMILES(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.cell_lines = torch.tensor(data['cell_line_encoded'].values, dtype=torch.long)
        self.drug_smiles = data['drug_smiles'].values
        self.targets = torch.tensor(data['ln_ic50'].values, dtype=torch.float)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'cell_line': self.cell_lines[idx],
            'drug_smiles': self.drugs_smiles[idx],
            'target': self.targets[idx]
        }

csv_file = "data/GDSC2_cleaned.csv"
dataset = DrugResponseDataset(csv_file)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model parameters
num_cell_lines = dataset.cell_lines.max().item() + 1
num_drugs = dataset.drugs.max().item() + 1
embed_dim = 128
hidden_dim = 64
output_dim = 1

# Initialize model, loss, optimizer
model = DrugResponseTransformer(num_cell_lines, num_drugs, embed_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        cell_line = batch['cell_line']
        drug = batch['drug']
        target = batch['target']

        optimizer.zero_grad()
        output = model(cell_line, drug).squeeze()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

# Save model
torch.save(model.state_dict(), "experiments/best_model.pt")
