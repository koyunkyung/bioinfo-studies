from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
import safe
import torch

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
        
        # safe 문자열 언어 모델 입력으로 바꿔주기
        inputs = self.tokenizer(safe_list, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state
    



### 함수 작동 확인 테스트 케이스 ###
if __name__ == "__main__":

    combined_cell_line = ["Camptothecin:TOP1", "Vinblastine:Microtubule destabiliser", "Cisplatin:DNA crosslinker"]
    smiles_list = ["CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2)O", "N.N.Cl[Pt]Cl", "CC1=C(C=C(C=N1)Cl)NCC2=CC=C(S2)C(=O)NC(CC3CCCC3)C(=O)NC4CC4"]

    ## SAFE-GPT를 사용한 임베딩 ##
    safe_gpt_embedding = SafeEmbeddingGPT(lm_model_name="datamol-io/safe-gpt")
    embeddings = safe_gpt_embedding.safe(smiles_list)
    print(f"SAFE-GPT Embeddings Shape: {embeddings.shape}")
    print(f"SAFE-GPT Embeddings Tensor: \n{embeddings}")

        

            
        