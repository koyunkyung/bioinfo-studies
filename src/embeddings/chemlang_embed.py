from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
import safe
import selfies as sf
from selfies import encoder as selfies_encoder
from selfies import get_alphabet_from_selfies, split_selfies
import torch
import numpy as np

### Drug name (각각의 약물에 해당되는 SMILES 정보 이용) ###

## 화학정보 임베딩 방식 통합 클래스 ##
class ChemicalEmbeddings:
    def __init__(self, safe_model_name="datamol-io/safe-gpt"):
        self.safe_embedding = SafeEmbeddingGPT(model_name=safe_model_name)
        self.selfies_embedding = SelfiesEmbeddingGPT()

    def get_safe_embeddings(self, smiles_list):
        return self.safe_embedding.embed(smiles_list)
    
    def get_selfies_embeddings(self, smiles_list):
        return self.selfies_embedding.embed(smiles_list)
    
# safe 언어 모델을 활용한 임베딩
class SafeEmbeddingGPT:
    def __init__(self, lm_model_name="datamol-io/safe-gpt"):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
        self.model = AutoModel.from_pretrained(lm_model_name)

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
        
        # safe 문자열 언어 모델 입력으로 바꿔주고 임베딩 추출
        inputs = self.tokenizer(safe_list, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state
    
# selfies 언어 모델을 활용한 임베딩
class SelfiesEmbeddingGPT:
    def __init__(self):
        # Robust alphabet 초기화
        self.robust_alphabet = sf.get_semantic_robust_alphabet()
        self.robust_alphabet.update(["[Pt]", "[nop]", "."])  # Custom symbols 추가
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






### 함수 작동 확인 테스트 케이스 ###
if __name__ == "__main__":

    combined_cell_line = ["Camptothecin:TOP1", "Vinblastine:Microtubule destabiliser", "Cisplatin:DNA crosslinker"]
    smiles_list = ["CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2)O", "CN(C)CC=CC(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OC4CCOC4", "CNC(=O)C1=CC=CC=C1SC2=CC3=C(C=C2)C(=NN3)C=CC4=CC=CC=N4"]
    ## scGPT를 사용한 cell line 임베딩 ##



    # ## SAFE-GPT를 사용한 약물 임베딩 ##
    # safe_gpt_embedding = SafeEmbeddingGPT(lm_model_name="datamol-io/safe-gpt")
    # embeddings = safe_gpt_embedding.safe(smiles_list)
    # print(f"SAFE-GPT Embeddings Shape: {embeddings.shape}")
    # print(f"SAFE-GPT Embeddings Tensor: \n{embeddings}")


    ## selfies-GPT를 사용한 약물 임베딩 ##
    selfies_gpt_embedding = SelfiesEmbeddingGPT()
    embeddings = selfies_gpt_embedding.embed(smiles_list)
    print(f"Selfies-GPT Embeddings Shape: {embeddings.shape}")
    print(f"Selfies-GPT Embeddings Tensor: \n{embeddings}")




    # ## MPNN을 사용한 약물 임베딩 ##
    # atom_dim = 6
    # bond_dim = 4 
    # batch_size = 32
    # message_units = 64
    # message_steps = 4
    # num_attention_heads = 8
    # dense_units = 512

    # mpnn_model = MPNNModel(atom_dim=atom_dim, bond_dim=bond_dim, batch_size=batch_size, message_units=message_units, message_steps=message_steps, num_attention_heads=num_attention_heads,dense_units=dense_units,)
    # embeddings = mpnn_embeddings(smiles_list, mpnn_model, batch_size=batch_size)

    # print(f"Generated Embeddings Shape: {embeddings.shape}")
    # print(f"Generated Embeddings: \n{embeddings}")
        

            
        