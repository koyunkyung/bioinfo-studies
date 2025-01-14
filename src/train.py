import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from embeddings import *

### load dataset ###

class DrugResponseData(Dataset):

    def __init__(self, combined_cell_line, combined_drug, ln_ic50, cell_embeddings_method, drug_embeddings_method):
        self.combined_cell_line = combined_cell_line
        self.combined_drug = combined_drug
        self.ln_ic50 = ln_ic50
        self.cell_embeddings_method = cell_embeddings_method
        self.drug_embeddings_method = drug_embeddings_method

        self.cell_line_embedding = CellLineEmbedding()
        self.cell_line_embedding = DrugEmbedding()

        


