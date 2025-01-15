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
import selfies as sf
import safe

### Cell line name ###
class CellLineEmbedding:
    def __init__(self, scbert_model_path, num_bins=7, dim=200, depth=6, seq_len=16906, heads=10):
        
        # !!!'model.pth' 못 찾아내서 구현 실해한 상태 재시도 필요 !!!
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
        pass

    # 1. Fingerprint 분자구조 임베딩
    class Fingerprint:

        # Morgan FP
        def morganFP(self, smiles_list, radius=2, n_bits=2048):
            fingerprints = []
            generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = generator.GetFingerprint(mol)
                    fingerprints.append(np.array(fp))
                else:
                    print(f"Invalid SMILES string: {smiles}")
            fingerprints_array = np.array(fingerprints)
            return torch.tensor(fingerprints_array, dtype=torch.float32)
            
        
        # ECFP (Extended Connectivity Fingerprints)
        def ECFP(self, smiles_list):
            return self.morganFP(smiles_list, radius=2)
        
    # 2. GNN (Graph Neural Network)
    def neural_graph():
        pass

    # 3. SMILES 파생 임베딩 기법: SELFIES, SAFE
    class fromSMILES():

        # SELFIES (Self-Referencing Embedded Strings)
        # !!! SELFIES Embeddings Shape: torch.Size([2]) - 수정 필요함 !!!
        def selfies(self, smiles_list):
            selfies_list = [sf.encoder(smiles) for smiles in smiles_list]
            # robust alphabet 생성하고 고유 ID로 매핑
            robust_alphabet = sf.get_semantic_robust_alphabet()
            robust_alphabet.update(["[Pt]", "[nop]", "."]) 
            alphabet = list(sorted(robust_alphabet))
            symbol_to_idx = {s: i for i, s in enumerate(alphabet)}

            pad_to_len = max(sf.len_selfies(s) for s in selfies_list)
            selfies_encoded = [
                sf.selfies_to_encoding(
                    selfies=s,
                    vocab_stoi=symbol_to_idx,
                    pad_to_len=pad_to_len,
                    enc_type="label",
                )[0]
                for s in selfies_list
            ]
            selfies_array = np.array(selfies_encoded, dtype=np.int32)
            return torch.tensor(selfies_array, dtype=torch.long)

        # SAFE (Sequential Attachment-based Fragment Embedding)

        def safe(self, smiles_list):
            # 유효성 검사 후에 safe로 변환
            safe_list = []
            for smiles in smiles_list:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        print(f"Invalid SMILES: {smiles}")
                        continue
                    safe_encoded = safe.encode(smiles)
                    safe_list.append(safe_encoded)
                except safe._exception.SAFEFragmentationError as e:
                    print(f"SAFE encoding failed for SMILES: {smiles} with error: {e}")
                    continue
            if not safe_list:
                raise ValueError("No valid SAFE encodings generated from the provided SMILES")

            tokenizer = AutoTokenizer.from_pretrained("datamol-io/safe-gpt")
            model = AutoModel.from_pretrained("datamol-io/safe-gpt")
            model.eval()
            safe_tokens = tokenizer(safe_list, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**safe_tokens)
            return outputs.pooler_output



### 함수 작동 확인 테스트 케이스 ###
if __name__ == "__main__":

    combined_cell_line = ["Camptothecin:TOP1", "Vinblastine:Microtubule destabiliser", "Cisplatin:DNA crosslinker"]
    smiles_list = ["CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2)O", "N.N.Cl[Pt]Cl", "CCC1(CC2CC(C3=C(CCN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)C78CCN9C7C(C=CC9)(C(C(C8N6C)(C(=O)OC)O)OC(=O)C)CC)OC)C(=O)OC)O"]

    cell_embedding = CellLineEmbedding(scbert_model_path="data/pretrained_models/scbert_pretrained.pth")

    drug_embedding = DrugEmbedding()
    fingerprint = drug_embedding.Fingerprint()
    fromsmiles = drug_embedding.fromSMILES()

    # # scBERT embedding
    # scbert_embeddings = cell_embedding.scBERT(combined_cell_line)
    # print(f"scBERT Embeddings Shape: {scbert_embeddings.shape}")


    # bioBERT embedding
    biobert_embeddings = cell_embedding.bioBERT(combined_cell_line)
    print(f"BioBERT Embeddings Shape: {biobert_embeddings.shape}")
    print(f"BioBERT Embeddings Tensor: \n{biobert_embeddings}")


    # Morgan fingerprint embedding
    morgan_embeddings = fingerprint.morganFP(smiles_list)
    print(f"Morgan Fingerprint Embeddings Shape: {morgan_embeddings.shape}")
    print(f"Morgan Fingerprint Embeddings Tensor: \n{morgan_embeddings}")


    # ECFP embedding
    ecfp_embeddings = fingerprint.ECFP(smiles_list)
    print(f"ECFP Embeddings Shape: {ecfp_embeddings.shape}")
    print(f"ECFP Embeddings Tensor: \n{ecfp_embeddings}")

    # SELFIES embedding
    selfies_embeddings = fromsmiles.selfies(smiles_list)
    print(f"SELFIES Embeddings Shape: {selfies_embeddings.shape}")
    print(f"SELFIES Embeddings Tensor: {selfies_embeddings}")


    # SAFE embedding
    safe_embeddings = fromsmiles.safe(smiles_list)
    print(f"SAFE Embeddings Shape: {safe_embeddings.shape}")
    print(f"SAFE Embeddings Tensor: {safe_embeddings}")


