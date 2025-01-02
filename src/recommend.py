import torch
from model import DrugResponseTransformer
import pandas as pd

# Load processed data and model
processed_file = "data/GDSC2_processed.csv"
model_path = "experiments/best_model.pt"

# Load dataset
data = pd.read_csv(processed_file)

# Load encoders
cell_line_classes = pd.read_csv("data/cell_line_classes.txt", header=None).squeeze().tolist()
drug_classes = pd.read_csv("data/drug_classes.txt", header=None).squeeze().tolist()

# Model parameters
num_cell_lines = len(cell_line_classes)
num_drugs = len(drug_classes)
embed_dim = 128
hidden_dim = 64
output_dim = 1

# Load model
model = DrugResponseTransformer(num_cell_lines, num_drugs, embed_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load(model_path))
model.eval()

# Recommend function
def recommend_best_drug(cell_line_name):
    if cell_line_name not in cell_line_classes:
        raise ValueError(f"Cell line '{cell_line_name}' not found in the dataset.")

    cell_line_index = cell_line_classes.index(cell_line_name)

    # Generate predictions for all drugs
    cell_line_tensor = torch.tensor([cell_line_index] * num_drugs, dtype=torch.long)
    drug_tensor = torch.arange(num_drugs, dtype=torch.long)

    with torch.no_grad():
        predictions = model(cell_line_tensor, drug_tensor).squeeze()

    # Find the best drug
    best_drug_index = predictions.argmin().item()
    best_drug_name = drug_classes[best_drug_index]
    best_prediction = predictions[best_drug_index].item()
    putative_target = data.loc[data["drug_encoded"] == best_drug_index, "putative_target"].values[0]

    return best_drug_name, best_prediction, putative_target

# Example usage
cell_line_input = "A375"  # Example cell line
try:
    best_drug, prediction, target = recommend_best_drug(cell_line_input)
    print(f"The best drug for cell line '{cell_line_input} is:")
    df = pd.DataFrame({
        "Drug Name": [best_drug],
        "Predicted ln_ic50": [round(prediction, 4)],
        "Putative Target": [target]
    })
    print(df)
except ValueError as e:
    print(e)

## Visualization ##
import matplotlib.pyplot as plt

def visualize_recommendation(drug_name, prediction, target):
    plt.figure(figsize=(10, 6))
    plt.barh([drug_name], [prediction], color="skyblue")
    plt.xlabel("Predicted ln_ic50")
    plt.ylabel("Drug")
    plt.title(f"Best Drug Recommendation\nPutative Target: {target}")
    plt.tight_layout()
    plt.show()

# Example usage with visualization
try:
    best_drug, prediction, target = recommend_best_drug(cell_line_input)
    visualize_recommendation(best_drug, prediction, target)
except ValueError as e:
    print(e)


def recommend_top_n_drugs(cell_line_name, n=5):
    if cell_line_name not in cell_line_classes:
        raise ValueError(f"Cell line '{cell_line_name}' not found in the dataset.")

    cell_line_index = cell_line_classes.index(cell_line_name)

    # Generate predictions for all drugs
    cell_line_tensor = torch.tensor([cell_line_index] * num_drugs, dtype=torch.long)
    drug_tensor = torch.arange(num_drugs, dtype=torch.long)

    with torch.no_grad():
        predictions = model(cell_line_tensor, drug_tensor).squeeze().tolist()

    # Get top n drugs
    top_indices = sorted(range(len(predictions)), key=lambda i: predictions[i])[:n]
    top_drugs = [drug_classes[i] for i in top_indices]
    top_targets = [data.loc[data["drug_encoded"] == i, "putative_target"].values[0] for i in top_indices]
    top_predictions = [predictions[i] for i in top_indices]

    return pd.DataFrame({
        "Drug Name": top_drugs,
        "Predicted ln_ic50": top_predictions,
        "Putative Target": top_targets
    })

# Visualize top n drugs
try:
    top_drugs_df = recommend_top_n_drugs(cell_line_input, n=5)
    print(top_drugs_df)

    # Bar plot for top n drugs
    plt.figure(figsize=(12, 6))
    plt.barh(top_drugs_df["Drug Name"], top_drugs_df["Predicted ln_ic50"], color="skyblue")
    plt.xlabel("Predicted ln_ic50")
    plt.ylabel("Drug")
    plt.title(f"Top 5 Drug Recommendations for Cell Line: {cell_line_input}")
    plt.tight_layout()
    plt.show()
except ValueError as e:
    print(e)
