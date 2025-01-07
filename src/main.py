import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from train import train_model
from evaluate import evaluate_model
from model import UnifiedTransformer
from train import GDSCDataset


# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, delta=0.001, save_path="best_model.pth"):
        """
        Early stopping to terminate training when validation loss doesn't improve.
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
       if self.best_loss is None or val_loss < self.best_loss - self.delta:
          # new best model found, reset counter and save model
          self.best_loss = val_loss
          self.counter = 0
          torch.save(model.state_dict(), self.save_path)
          
       else:
          # no improvement
          self.counter += 1
          if self.counter >= self.patience:
             self.early_stop = True


if __name__ == "__main__":
    CSV_PATH = "data/processed/GDSC2_embeddings.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    PATIENCE = 10  # Number of epochs to wait before stopping

    dataset = GDSCDataset(CSV_PATH)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize the model
    model = UnifiedTransformer(
        embedding_dim_cell=1,  # cell_line_name_encoded has 1 dimension
        embedding_dim_drug=128,
        hidden_dim=256,
        output_dim=1,
        drug_embedding_type="scbert",  # Change to "scbert" if using SCBERT inputs
        rdkit_feature_dim=4  # RDKit features: MolWt, LogP, NumHDonors, NumHAcceptors
    ).to(DEVICE)

    # Define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=PATIENCE, save_path="experiments/best_model.pth")

    # Training and evaluation loop
    for epoch in range(EPOCHS):
        # Train the model
        train_loss = train_model(model, data_loader, optimizer, criterion, DEVICE)

        # Evaluate the model
        val_loss, rmse, r2 = evaluate_model(model, data_loader, criterion, DEVICE)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")

        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Restoring best model.")
            model.load_state_dict(torch.load("best_model.pth"))
            break

    print("Training complete.")

