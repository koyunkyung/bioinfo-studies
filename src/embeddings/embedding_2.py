from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
import safe
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
        self.alphabet = None
        self.alphabet_dict = None
    
    def initialize_alphabet(self, smiles_list):
        # selfies 패키지에서 제공하는 encoder를 사용해 smiles를 selfies로 변환
        selfies_list = [selfies_encoder(smiles) for smiles in smiles_list]
        self.alphabet = sorted(list(get_alphabet_from_selfies("".join(selfies_list))))
        self.alphabet_dict = {symbol: idx for idx, symbol in enumerate(self.alphabet)}

    def selfies_onehot(self, selfies_str):
        symbols = split_selfies(selfies_str)
        one_hot = np.zeros((len(symbols), len(self.alphabet)), dtype=np.float32)

        for i, symbol in enumerate(symbols):
            if symbol in self.alphabet_dict:
                one_hot[i, self.alphabet_dict[symbol]] = 1.0
            else:
                raise ValueError(f"Symbol {symbol} not in alphabet.")
        return one_hot
    
    def embed(self, smiles_list):
        if self.alphabet is None:
            self.initialize_alphabet(smiles_list)

        embeddings = []
        for smiles in smiles_list:
            selfies_str = selfies_encoder(smiles)
            one_hot = self.selfies_onehot(selfies_str)
            embeddings.append(one_hot)

        return embeddings




### 함수 작동 확인 테스트 케이스 ###
if __name__ == "__main__":

    combined_cell_line = ["Camptothecin:TOP1", "Vinblastine:Microtubule destabiliser", "Cisplatin:DNA crosslinker"]
    smiles_list = ["CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2)O", "N.N.Cl[Pt]Cl", "CC1=C(C=C(C=N1)Cl)NCC2=CC=C(S2)C(=O)NC(CC3CCCC3)C(=O)NC4CC4"]

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
        

            
        