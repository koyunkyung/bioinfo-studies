import torch
import pandas as pd

### Custom Dataset ###
class GDSCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):

        data = pd.read_csv(csv_path)
        # cell line name embeddings
        self.cell_line_inputs = torch.tensor(
            data[['cell_line_name_encoded']].values, dtype=torch.float32
        )
        # drug name inputs (e.g. RDKit features or tokenized SCBERT inputs)
        if 'MolWt' in data.columns:
            self.drug_inputs = torch.tensor(
                data[['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors']].values, dtype=torch.float32
            )
        else:
            raise ValueError("Drug input data not found.")
        # target values
        self.targets = torch.tensor(data['ln_ic50'].values, dtype=torch.float32)

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
        # move data to device
        cell_line_inputs = batch['cell_line_inputs'].to(device)
        drug_inputs = batch['drug_inputs'].to(device)
        targets = batch['target'].to(device)

        # zero gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(cell_line_inputs, drug_inputs).squeeze()
        loss = criterion(outputs, targets)

        # backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(targets)

    epoch_loss = running_loss / len(data_loader.dataset)
    return epoch_loss
