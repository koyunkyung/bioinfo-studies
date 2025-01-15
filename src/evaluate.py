import torch
from sklearn.metrics import mean_squared_error, r2_score
from embeddings import CellLineEmbedding, DrugEmbedding
from model import DrugCellTransformer
from train import *

model = DrugCellTransformer

model.eval()
predictions = []
true_values = []

with torch.no_grad():
    dataset = DrugResponseData(combined_cell_name, drug_smiles, labels, cell_embedding_method, drug_embedding_method)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in dataloader:
        cell_features = batch['cell'].squeeze(1)
        drug_features = batch['drug'].squeeze(1)
        labels = batch['label']

        outputs = model(drug_features, cell_features).squeeze(1)
        predictions.extend(outputs.tolist())
        true_values.extend(labels.tolist())

mse = mean_squared_error(true_values, predictions)
r2 = r2_score(true_values, predictions)

print(f"MSE: {mse}, R2: {r2}")
