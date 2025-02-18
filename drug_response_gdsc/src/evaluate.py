import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from model import DrugResponsePredictor
from train import DrugResponseDataset

def load_evaluation_data(file_path):
    data = pd.read_csv(file_path)
    cell_lines = data["cell_line_name"].tolist()
    diseases = data["disease"].tolist()
    drugs = data["smiles"].tolist()
    drug_names = data["drug_name"].tolist()
    targets = data["putative_target"].tolist()
    ln_ic50 = data["ln_ic50"].tolist()
    return cell_lines, diseases, drugs, drug_names, targets, ln_ic50

def evaluate(model, dataloader, device):
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for cell_line_emb, drug_emb, ln_ic50 in dataloader:
            cell_line_emb = cell_line_emb.to(device)
            drug_emb = drug_emb.to(device)
            ln_ic50 = ln_ic50.to(device)

            predictions = model(cell_line_emb, drug_emb).squeeze()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(ln_ic50.cpu().numpy())

    mse = mean_squared_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    return mse, r2

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_file = "data/processed/GDSC2_cleaned.csv"
    cell_lines, diseases, drugs, drug_names, targets, ln_ic50 = load_evaluation_data(data_file)

    dataset = DrugResponseDataset(
        cell_lines, diseases, drugs, drug_names, targets, ln_ic50, drug_embedding_method="one_hot"
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Dynamically adjust dimensions
    drug_embedding_dim = dataset.one_hot_vector_size
    cell_line_embedding_dim = 100  # Adjust based on your configuration

    # Initialize model with matching dimensions
    model = DrugResponsePredictor(cell_line_embedding_dim, drug_embedding_dim)
    model.to(device)
    model.eval()

    # Evaluate
    mse, r2 = evaluate(model, dataloader, device)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R² Score: {r2}")
