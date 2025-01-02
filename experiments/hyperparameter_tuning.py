import optuna
import torch
from torch.utils.data import DataLoader

import sys
import os
# Add 'src' directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "../src")
sys.path.append(src_dir)

from model import DrugResponseTransformer
from train import DrugResponseDataset
from sklearn.metrics import mean_squared_error, r2_score
import torch.nn as nn
import torch.optim as optim


def objective(trial):
    # Load dataset
    dataset = DrugResponseDataset("data/GDSC2_cleaned.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model parameters from trial
    embed_dim = trial.suggest_categorical("embed_dim", [64, 128, 256])
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)

    # Initialize model
    num_cell_lines = dataset.cell_lines.max().item() + 1
    num_drugs = dataset.drugs.max().item() + 1
    model = DrugResponseTransformer(num_cell_lines, num_drugs, embed_dim, hidden_dim, 1)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(5):  # Reduce epochs for tuning speed
        model.train()
        for batch in dataloader:
            cell_line = batch['cell_line']
            drug = batch['drug']
            target = batch['target']

            optimizer.zero_grad()
            output = model(cell_line, drug).squeeze()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Validation (assume validation data exists)
    model.eval()
    true_values, predictions = [], []
    with torch.no_grad():
        for batch in dataloader:
            cell_line = batch['cell_line']
            drug = batch['drug']
            target = batch['target']
            output = model(cell_line, drug).squeeze()
            predictions.extend(output.tolist())
            true_values.extend(target.tolist())

    # Evaluate RMSE
    rmse = mean_squared_error(true_values, predictions, squared=False)
    return rmse

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    print("Best trial:", study.best_trial.params)
