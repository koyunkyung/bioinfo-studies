import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import MolFromSmiles
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, BertTokenizer
from performer_pytorch import PerformerLM
import scanpy as sc
from selfies import encoder

### Cell line name ###
class CellLineEmbedding:
    def __init__(self, scbert_model_path, num_bins=7, dim=200, depth=6, seq_len=16906, heads=10):
        # # scBERT 관련 초기화
        # self.seq_len = seq_len
        # self.num_bins = num_bins
        # self.scbert_model = PerformerLM(
        #     num_tokens=num_bins + 2, 
        #     dim=dim,
        #     depth=depth,
        #     max_seq_len=seq_len + 1,
        #     heads=heads,
        #     local_attn_heads=0,
        #     g2v_position_emb=True,
        # )
        # ckpt = torch.load(scbert_model_path)
        # self.scbert_model.load_state_dict(ckpt["model_state_dict"])
        # self.scbert_model.eval()


        # bioBERT 관련 초기화
        self.biobert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        self.biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

    # scBERT에서 combined_cell_line 임베딩하기 위해 토큰화 먼저 진행
    def tokenize(self, combined_cell_line):
        vocab = {char: idx for idx, char in enumerate(set("".join(combined_cell_line)))}
        tokenized_data = []
        for entry in combined_cell_line:
            tokens = [vocab[char] if char in vocab else 0 for char in entry]
            # 입력 시퀀스 크기 맞추기 (패딩)
            tokens = tokens[:self.seq_len] + [0] * (self.seq_len - len(tokens))
            tokenized_data.append(tokens)
        tokenized_tensor = torch.tensor(tokenized_data, dtype=torch.long)
        # 특수 토큰 추가
        tokenized_tensor = torch.cat(
            (tokenized_tensor, torch.zeros((len(combined_cell_line), 1), dtype=torch.long)), dim=1
        )
        return tokenized_tensor
    

    # 1. scBERT (single cell BERT)

    def scBERT(self, combined_cell_line):  
        tokenized_input = self.tokenize(combined_cell_line)
        with torch.no_grad():
            outputs = self.scbert_model(tokenized_input)
        return outputs

    # 2. bioBERT
    def bioBERT(self, combined_cell_line):
        inputs = self.biobert_tokenizer(combined_cell_line, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.biobert_model(**inputs)
        return outputs.pooler_output


### Drug name (분자구조 활용한 GNN 기법들 사용)###
class DrugEmbedding:

    def __init__(self):
        raise NotImplementedError

    # 1. Fingerprint 분자구조 임베딩
    class Fingerprint:

        # Morgan FP
        def morganFP(self, smiles_list, radius=2, n_bits=2048):
            fingerprints = []
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fingerprints.append(np.array(fp))
            return torch.tensor(fingerprints, dtype=torch.float32)
            
        
        # ECFP (Extended Connectivity Fingerprints)
        def ECFP(self, smiles_list):
            return self.morganFP(smiles_list, radius=2)
        
    # 2. GNN (Graph Neural Network)
    def graph():
        raise NotImplementedError

    # 3. SMILES 파생 임베딩 기법: SELFIES, SAFE
    class fromSMILES():

        # SELFIES (Self-Referencing Embedded Strings)
        def selfies(self, smiles_list):
            return encoder(smiles_list)
        
        # SAFE (Sequential Attachment-based Fragment Embedding)
        def safe():
            raise NotImplementedError
        

### 함수 작동 확인 테스트 케이스 ###
if __name__ == "__main__":
    combined_cell_line = ["Camptothecin:TOP1", "Vinblastine:Microtubule destabiliser", "Cisplatin:DNA crosslinker"]
    smiles_list = ["CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2)O", "N.N.Cl[Pt]Cl"]

    cell_embedding = CellLineEmbedding(scbert_model_path="data/pretrained_models/scbert_pretrained.pth")
    drug_embedding = DrugEmbedding()

    # # scBERT embedding
    # scbert_embeddings = cell_embedding.scBERT(combined_cell_line)
    # print(f"scBERT Embeddings Shape: {scbert_embeddings.shape}")


    # bioBERT embedding
    biobert_embeddings = cell_embedding.bioBERT(combined_cell_line)
    print(f"BioBERT Embeddings Shape: {biobert_embeddings.shape}")

    # Morgan fingerprint embedding
    morgan_embeddings = drug_embedding.Fingerprint.morganFP(smiles_list)
    print(f"Morgan Fingerprint Embeddings Shape: {morgan_embeddings.shape}")

    # ECFP embedding
    ecfp_embeddings = drug_embedding.Fingerprint.ECFP(smiles_list)
    print(f"ECFP Embeddings Shape: {ecfp_embeddings.shape}")

