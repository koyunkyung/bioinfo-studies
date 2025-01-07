import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on validation/test set.
    """
    model.eval()
    running_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in data_loader:
            # Move data to device
            cell_line_inputs = batch['cell_line_inputs'].to(device)
            drug_inputs = batch['drug_inputs'].to(device)
            targets = batch['target'].to(device)

            # Forward pass
            outputs = model(cell_line_inputs, drug_inputs).squeeze()

            # Compute loss
            loss = criterion(outputs, targets)

            running_loss += loss.item() * len(targets)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    # Calculate metrics
    epoch_loss = running_loss / len(data_loader.dataset)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)

    return epoch_loss, rmse, r2