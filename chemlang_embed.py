from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, GPT2TokenizerFast
from rdkit import Chem
import safe
import selfies as sf
from selfies import encoder as selfies_encoder
from selfies import get_alphabet_from_selfies, split_selfies
import torch
import numpy as np
from data.safe_gpt.tokenizer import *
from data.safe_gpt.model import *

### Drug name (각각의 약물에 해당되는 SMILES 정보 이용) ###
## 화학정보 임베딩 방식 통합 클래스 ##
class ChemicalEmbeddings:
    def __init__(self):
        self.safe_embedding = SafeEmbeddingGPT()
        self.selfies_embedding = SelfiesEmbeddingGPT()
        self.chemberta_embedding = ChemBertaEmbedding()

    def get_safe_embeddings(self, smiles_list):
        return self.safe_embedding.embed(smiles_list)
    
    def get_selfies_embeddings(self, smiles_list):
        return self.selfies_embedding.embed(smiles_list)
    
    def get_chemberta_embeddings(self, smiles_list):
        return self.chemberta_embedding.embed(smiles_list)
    
# safe 언어 모델을 활용한 약물 임베딩
class SafeEmbeddingGPT:
    def __init__(self):
        try:
           self.model = SAFEDoubleHeadsModel.from_pretrained("datamol-io/safe-gpt")
           self.tokenizer = SAFETokenizer.from_pretrained("datamol-io/safe-gpt")
        except Exception as e:
           print(f"Error initializing SafeEmbeddingGPT: {e}")
           raise

    def prepare_inputs(self, smiles_list):
        encoded_inputs = [self.tokenizer.encode(smiles) for smiles in smiles_list]

        max_len = max(len(ids) for ids in encoded_inputs)
        padded_inputs = []
        attention_masks = []
        for ids in encoded_inputs:
            padding_length = max_len - len(ids)
            padded_sequence = ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * len(ids) + [0] * padding_length
            padded_inputs.append(padded_sequence)
            attention_masks.append(attention_mask)
        return {
            'input_ids': torch.tensor(padded_inputs, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        }

    def embed(self, smiles_list):
        inputs = self.prepare_inputs(smiles_list)
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs[0]
            embeddings = last_hidden[:,0,:]    # cls 토큰 추출: beginning, "summary" position of the sentence
        return embeddings

# selfies 언어 모델을 활용한 임베딩
class SelfiesEmbeddingGPT:
    def __init__(self):
        # robust alphabet 초기화
        self.robust_alphabet = sf.get_semantic_robust_alphabet()
        self.robust_alphabet.update(["[Pt]", "[nop]", "."])
        self.alphabet = list(sorted(self.robust_alphabet))
        self.symbol_to_idx = {s: i for i, s in enumerate(self.alphabet)}

    def validate_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None

    def selfies(self, smiles_list):
        # SMILES를 SELFIES로 변환
        valid_smiles = [smiles for smiles in smiles_list if self.validate_smiles(smiles)]
        selfies_list = []
        for smiles in valid_smiles:
            try:
                selfies_str = sf.encoder(smiles)
                selfies_list.append(selfies_str)
            except Exception as e:
                print(f"Error encoding SMILES '{smiles}': {e}")

        if not selfies_list:
            raise ValueError("No valid SELFIES strings generated.")

        pad_to_len = max(sf.len_selfies(s) for s in selfies_list)

        # SELFIES를 정수로 인코딩
        selfies_encoded = [
            sf.selfies_to_encoding(
                selfies=s,
                vocab_stoi=self.symbol_to_idx,
                pad_to_len=pad_to_len,
                enc_type="label",
            )[0]
            for s in selfies_list
        ]
        selfies_array = np.array(selfies_encoded, dtype=np.int32)
        return torch.tensor(selfies_array, dtype=torch.long)

    def embed(self, smiles_list):
        try:
            embeddings = self.selfies(smiles_list)
            return embeddings
        except ValueError as e:
            print(f"Error generating embeddings: {e}")
            return None

# ChemBERTa를 이용한 약물 임베딩
class ChemBertaEmbedding:
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

    def embed(self, smiles_list):
        inputs = self.tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits 
        cls_embeddings = logits[:, 0, :] 
        return cls_embeddings



### 함수 작동 확인 테스트 케이스 ###
if __name__ == "__main__":
    combined_cell_line = ["Camptothecin:TOP1", "Vinblastine:Microtubule destabiliser", "Cisplatin:DNA crosslinker"]
    smiles_list = ["CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2)O", "CN(C)CC=CC(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OC4CCOC4", "CNC(=O)C1=CC=CC=C1SC2=CC3=C(C=C2)C(=NN3)C=CC4=CC=CC=N4"]

    embedder = ChemicalEmbeddings()
    
    # SAFE-GPT 임베딩 테스트
    safe_embeddings = embedder.get_safe_embeddings(smiles_list)
    print(f"SAFE-GPT Embeddings Shape: {safe_embeddings.shape}")
    print(f"SAFE-GPT Embeddings Tensor:\n{safe_embeddings}\n")

    # # SELFIES 임베딩 테스트
    # selfies_embeddings = embedder.get_selfies_embeddings(smiles_list)
    # print(f"Selfies-GPT Embeddings Shape: {selfies_embeddings.shape}")
    # print(f"Selfies-GPT Embeddings Tensor:\n{selfies_embeddings}\n")

    # # ChemBERTa 임베딩 테스트
    # chemberta_embeddings = embedder.get_chemberta_embeddings(smiles_list)
    # print(f"ChemBERTa Embeddings Shape: {chemberta_embeddings.shape}")
    # print(f"ChemBERTa Embeddings Tensor:\n{chemberta_embeddings}")