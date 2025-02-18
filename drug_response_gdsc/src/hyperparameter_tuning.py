import optuna
import torch
from torch.utils.data import DataLoader
from train import DrugResponseDataset, train_model  # Import your dataset and training function
from model import DrugResponsePredictor  # Import your model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Objective function for Optuna
def objective(trial):
    # Define the search space for hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    num_hidden_units = trial.suggest_int("num_hidden_units", 64, 512)
    patience = trial.suggest_int("patience", 5, 20)

    # Load data
    data = pd.read_csv("data/processed/GDSC2_cleaned.csv")
    # Extract required columns
    cell_lines = data["cell_line_name"].tolist()
    diseases = data["disease"].tolist()
    drugs = data["smiles"].tolist()
    drug_names = data["drug_name"].tolist()
    targets = data["putative_target"].tolist()
    ln_ic50 = data["ln_ic50"].tolist()

    # Split the data into train and validation sets
    train_cell_lines, val_cell_lines, train_drugs, val_drugs, train_drug_names, val_drug_names, train_targets, val_targets, train_ln_ic50, val_ln_ic50 = train_test_split(
        cell_lines, drugs, drug_names, targets, ln_ic50, test_size=0.2, random_state=42
    )

    # Create datasets and dataloaders
    train_dataset = DrugResponseDataset(
        train_cell_lines, train_drugs, train_drug_names, train_targets, train_ln_ic50, drug_embedding_method="chemberta"
    )
    val_dataset = DrugResponseDataset(
        val_cell_lines, val_drugs, val_drug_names, val_targets, val_ln_ic50, drug_embedding_method="chemberta"
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model initialization
    model = DrugResponsePredictor(
        cell_line_embedding_dim=100,
        drug_embedding_dim=768,
        num_hidden_units=num_hidden_units,
        dropout_rate=dropout_rate
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # Train the model
    train_model(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=50,
        patience=patience,
        device=device
    )

    # Evaluate on validation set
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for cell_line_emb, drug_emb, ln_ic50 in val_loader:
            cell_line_emb = cell_line_emb.to(device)
            drug_emb = drug_emb.to(device)
            ln_ic50 = ln_ic50.to(device)

            predictions = model(cell_line_emb, drug_emb).squeeze()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(ln_ic50.cpu().numpy())

    mse = mean_squared_error(all_targets, all_predictions)
    return mse  # Optuna will minimize this value


# Main function to run hyperparameter tuning
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)  # Run 50 trials

    # Log the best parameters
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (MSE): {trial.value}")
    print(f"  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save the best parameters to a file
    with open("experiments/best_hyperparameters.txt", "w") as f:
        f.write(f"Best MSE: {trial.value}\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
