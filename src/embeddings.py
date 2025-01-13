import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from transformers import AutoTokenizer, AutoModel

### Cell line name ###
class CellLineEmbedding:
    def __init__(self, drug_name, disease):
        self.drug_name = drug_name
        self.disease = disease

    # 1. scBERT (single cell BERT)
    def scBERT():
        raise NotImplementedError
    

    # 2. bioBERT
    def bioBERT(self, drug_name):

        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

        inputs = tokenizer(drug_name)
        with torch.no_grad():
            outputs = model(**inputs)
        
        embeddings = 

### Drug name ###
# 1. Fingerprint 분자구조 임베딩
class Fingerprint:

    # Morgan FP
    def morganFP():
        raise NotImplementedError
    
    # ECFP (Extended Connectivity Fingerprints)
    def ECFP():
        raise NotImplementedError
    
# 2. GNN (Graph Neural Network)
def graph():
    raise NotImplementedError

# 3. SMILES 파생 임베딩 기법: SELFIES, SAFE
class fromSMILES():

    # SELFIES (Self-Referencing Embedded Strings)
    def selfies():
        raise NotImplementedError
    
    # SAFE (Sequential Attachment-based Fragment Embedding)
    def safe():
        raise NotImplementedError