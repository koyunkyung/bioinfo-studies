import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

### Custom Dataset ###
class GDSCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        
        # cell line name embeddings
        scaler = MinMaxScaler()
        self.cell_line_inputs = torch.tensor(
            scaler.fit_transform(data[['cell_line_name_encoded']]), dtype=torch.float32
        )

        # drug name inputs (e.g. RDKit features or tokenized SCBERT inputs)
        if 'MolWt' in data.columns:
            self.drug_inputs = torch.tensor(
                scaler.fit_transform(
                data[['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors']]),
                dtype=torch.float32
            )
        
        self.drug_inputs = data['smiles'].tolist()
        
        # target values
        self.targets = torch.tensor(
            scaler.fit_transform(data[['ln_ic50']]), dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'cell_line_inputs': self.cell_line_inputs[idx],
            'drug_inputs': self.drug_inputs[idx],
            'target': self.targets[idx]
        }

### Training Function ###
def train_model(model, data_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in data_loader:
        # Handle drug_inputs
        cell_line_inputs = batch['cell_line_inputs'].to(device)
        if isinstance(batch['drug_inputs'], list):  # For SCBERT (SMILES strings)
            drug_inputs = batch['drug_inputs']  # Keep as a list for tokenization in the model
        else:  # For RDKit molecular features
            drug_inputs = batch['drug_inputs'].to(device)

        targets = batch['target'].to(device)

        # zero gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(cell_line_inputs, drug_inputs).squeeze()
        loss = criterion(outputs, targets.squeeze())

        # backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(targets)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    scheduler.step(running_loss)

    epoch_loss = running_loss / len(data_loader.dataset)
    return epoch_loss
