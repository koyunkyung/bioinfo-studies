import torch
import json
from typing import List, Dict
from data.sc_gpt.model import *
from data.sc_gpt.gene_tokenizer import *

class scGPTEmbedder:
    def __init__(self, model_path, vocab_path):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vocab = SimpleGeneVocab.from_file(vocab_path)  # 유전자 이름이랑 고유ID 매핑된 파일 가져오기

        self.model = self.initialize_model()
        if model_path:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)            
        self.model = self.model.to(self.device)
        self.model.eval()

    # args.json 파일에서 기본 하이퍼파라미터 가져오기  
    def initialize_model(self):
        model_params = {
            "ntoken": len(self.vocab),
            "d_model": 512,
            "nhead": 8,
            "d_hid": 512,
            "nlayers": 12,
            "dropout": 0.1,
            "input_emb_style": "continuous",
            "n_input_bins": 51,
            "pad_token": "<pad>",
            "do_mvc": True,
            "pad_value": -2,
            "vocab": self.vocab,
            "use_fast_transformer": False 
        }
        return TransformerModel(**model_params)
        

    def process_cell_line(self, cell_line):
        if ":" in cell_line:
            cell, target = cell_line.split(":")
            return cell.strip(), target.strip()
        return cell_line.strip(), ""
    
    # 긱 cell line을 토큰 ID로 변환하기
    def tensor_input(self, cell_lines):
        processed = [self.process_cell_line(cl) for cl in cell_lines]
       
        token_ids = []
        for cell, target in processed:
            cell_id = self.vocab.get(cell, self.vocab["<pad>"])
            target_id = self.vocab.get(target, self.vocab["<pad>"])
            token_ids.append([self.vocab["<cls>"], cell_id, target_id])

        tokens_tensor = torch.tensor(token_ids).to(self.device)
        values_tensor = torch.zeros_like(tokens_tensor, dtype=torch.float)
        attention_mask = torch.zeros_like(tokens_tensor, dtype=torch.bool)

        return {
            "tokens": tokens_tensor,
            "values": values_tensor,
            "attention_mask": attention_mask
        }
    def get_embeddings(self, cell_lines):
        inputs = self.tensor_input(cell_lines)
        with torch.no_grad():
            outputs = self.model(
                src=inputs["tokens"],
                values=inputs["values"],
                src_key_padding_mask=inputs["attention_mask"],
                MVC=True    # scGPT에서 사용하는 특별한 학습 방식임 (모델이 유전자 패턴 더 잘 이해)
            )
        
        return outputs["cell_emb"]

    
if __name__ == "__main__":
    combined_cell_line = ["Camptothecin:TOP1", "Vinblastine:Microtubule destabiliser", "Cisplatin:DNA crosslinker"]
    smiles_list = ["CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2)O", "CN(C)CC=CC(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OC4CCOC4", "CNC(=O)C1=CC=CC=C1SC2=CC3=C(C=C2)C(=NN3)C=CC4=CC=CC=N4"]

    model_path = "./data/sc_gpt/best_model.pt"
    vocab_path = "./data/sc_gpt/vocab.json"

    embedder = scGPTEmbedder(model_path, vocab_path)
    embeddings = embedder.get_embeddings(combined_cell_line)

    print(f"scGPT Embeddings Shape: {embeddings.shape}")
    print(f"scGPT Embeddings Tensor:\n{embeddings}")