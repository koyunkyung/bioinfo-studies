import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import MolFromSmiles
from transformers import AutoTokenizer, AutoModel, BertTokenizer
from performer_pytorch import PerformerLM
import scanpy as sc

### Cell line name ###
class CellLineEmbedding:
    def __init__(self, cell_line_combined):
        self.cell_line_combined = cell_line_combined

    # 1. scBERT (single cell BERT)

    def scBERT(self, cell_line_combined):
        
        tokenizer = BertTokenizer.from_pretrained('')
        model =  PerformerLM(
            num_tokens = CLASS,
            dim = 200,
            depth = 6,
            max_seq_len = SEQ_LEN,
            heads = 10,
            local_attn_heads = 0,
            g2v_position_emb = False
        )

        inputs = tokenizer(cell_line_combined)
        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = 

    # 2. bioBERT
    def bioBERT(self, cell_line_combined):

        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

        inputs = tokenizer(cell_line_combined)
        with torch.no_grad():
            outputs = model(**inputs)
        
        embeddings = 

### Drug name (분자구조 활용한 GNN 기법들 사용)###
class DrugEmbedding:

    def __init__(self, drug_combined):
        self.drug_combined = drug_combined

    # 1. Fingerprint 분자구조 임베딩
    class Fingerprint:

        # Morgan FP
        def morganFP():
            Chem.RDKFingerprint()
        
        # ECFP (Extended Connectivity Fingerprints)
        def ECFP():
            AllChem.GetMorganFingerprint()
        
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
        

### 함수 작동 확인 테스트 케이스 ###
if __name__ == "__main__":
