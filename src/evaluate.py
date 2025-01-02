from model import DrugResponseTransformer
import torch
from torch.utils.data import DataLoader
from train import DrugResponseDataset
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load dataset and model
dataset = DrugResponseDataset("data/GDSC2_processed.csv")
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

num_cell_lines = dataset.cell_lines.max().item() + 1
num_drugs = dataset.drugs.max().item() + 1
model = DrugResponseTransformer(num_cell_lines, num_drugs, 128, 64, 1)
model.load_state_dict(torch.load("experiments/best_model.pt"))
model.eval()

# Evaluate
true_values, predictions = [], []
with torch.no_grad():
    for batch in dataloader:
        cell_line = batch['cell_line']
        drug = batch['drug']
        target = batch['target']
        output = model(cell_line, drug).squeeze()
        predictions.extend(output.tolist())
        true_values.extend(target.tolist())

# Metrics
rmse = mean_squared_error(true_values, predictions, squared=False)
r2 = r2_score(true_values, predictions)
print(f"RMSE: {rmse}, R2: {r2}")

# Scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(true_values, predictions, alpha=0.7, s=50)

x = np.linspace(min(true_values), max(true_values), 100)
plt.plot(x, x, '--', color='red', linewidth=2)

plt.xlabel('True ln_ic50', fontsize=14)
plt.ylabel('Predicted ln_ic50', fontsize=14)
plt.title('True vs Predicted ln_ic50', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()