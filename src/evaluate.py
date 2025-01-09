import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from train import DrugResponseDataset
from model import DrugResponsePredictor

def load_model(model_path, cell_line_embedding_dim, drug_embedding_dim, device):
    model = DrugResponsePredictor(cell_line_embedding_dim, drug_embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataloader, device):
    model.eval()
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

    # calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    return mse, r2

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv("data/processed/GDSC2_cleaned.csv")

    cell_lines = data["cell_line_name"].tolist()
    diseases = data["disease"].tolist()
    drugs = data["smiles"].tolist()
    drug_names = data["drug_name"].tolist()
    targets = data["putative_target"].tolist()
    ln_ic50 = data["ln_ic50"].tolist()

    embedding_methods = ["one_hot", "scbert", "chemberta"]
    results = {}

    for method in embedding_methods:
        print(f"\n=== Evaluating with {method} drug embeddings ===\n")

        # Create dataset and dataloader for the test set
        dataset = DrugResponseDataset(
            cell_lines, diseases, drugs, drug_names, targets, ln_ic50,
            drug_embedding_method=method
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        # Load model for the corresponding embedding method
        cell_line_embedding_dim = 100  # Adjust based on your Word2Vec embedding size
        if method == "one_hot":
            drug_embedding_dim = dataset.one_hot_vector_size
        elif method in ["scbert", "chemberta"]:
            drug_embedding_dim = 768
        else:
            raise ValueError(f"Unknown embedding method: {method}")

        model_path = f"models/{method}_model.pth"  # Path to the saved model
        model = load_model(model_path, cell_line_embedding_dim, drug_embedding_dim, device)

        # Evaluate the model
        mse, r2 = evaluate_model(model, dataloader, device)
        results[method] = {"MSE": mse, "R2": r2}
        print(f"{method} Embeddings - MSE: {mse:.4f}, R2: {r2:.4f}")

    # Log results
    print("\n=== Summary of Results ===")
    for method, metrics in results.items():
        print(f"{method} Embeddings - MSE: {metrics['MSE']:.4f}, R2: {metrics['R2']:.4f}")