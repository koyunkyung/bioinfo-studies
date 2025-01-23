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
    
# safe 언어 모델을 활용한 임베딩
class SafeEmbeddingGPT:
   def __init__(self):
       try:
           self.model = SAFEDoubleHeadsModel.from_pretrained("datamol-io/safe-gpt")
           self.tokenizer = SAFESplitter()
           self.vocab_size = self.model.config.vocab_size
       except Exception as e:
           print(f"Error initializing SafeEmbeddingGPT: {e}")
           raise

   def embed(self, smiles_list):
       # 유효성 검사 후에 safe로 변환 (safe 라이브러리 사용)
       safe_list = []
       for smiles in smiles_list:
           try:
               mol = Chem.MolFromSmiles(smiles)
               if mol is None:
                   print(f"Invalid SMILES: {smiles}")
                   continue     
               safe_encoded = safe.encode(smiles)
               if safe_encoded:
                   tokens = self.tokenizer.tokenize(safe_encoded)
                   safe_list.append(tokens)
           except Exception as e:
               print(f"Encoding failed for SMILES: {smiles} with error: {e}")
               continue
               
       if not safe_list:
           raise ValueError("No valid SAFE encodings generated from the provided SMILES")

       # 토큰을 모델 입력으로 변환 (해시 함수 이용해 모델의 어휘 크기에 맞는 정수 ID로 매핑)
       input_tensors = []
       for tokens in safe_list:
           token_ids = []
           for token in tokens:
               token_id = hash(token) % (self.vocab_size - 1)
               token_ids.append(token_id)
           input_tensors.append(torch.tensor(token_ids))

       # 가장 긴 sequence 기준으로 나머지에 padding 토큰 추가해서 길이 맞추기
       max_len = max(len(t) for t in input_tensors)
       padded_tensors = [torch.cat([t, torch.full((max_len - len(t),), self.vocab_size-1)]) for t in input_tensors]
       input_ids = torch.stack(padded_tensors)

        # 임베딩 추출
       with torch.no_grad():
           outputs = self.model(input_ids, output_hidden_states=True)
       return outputs.hidden_states[-1]


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

    # SELFIES 임베딩 테스트
    selfies_embeddings = embedder.get_selfies_embeddings(smiles_list)
    print(f"Selfies-GPT Embeddings Shape: {selfies_embeddings.shape}")
    print(f"Selfies-GPT Embeddings Tensor:\n{selfies_embeddings}\n")

    # ChemBERTa 임베딩 테스트
    chemberta_embeddings = embedder.get_chemberta_embeddings(smiles_list)
    print(f"ChemBERTa Embeddings Shape: {chemberta_embeddings.shape}")
    print(f"ChemBERTa Embeddings Tensor:\n{chemberta_embeddings}")