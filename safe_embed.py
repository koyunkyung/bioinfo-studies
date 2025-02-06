import safe
from safe.converter import *
from safe.tokenizer import SAFETokenizer
import torch


class SafeEmbeddingGPT:
    def __init__(self):
       self.tokenizer = SAFETokenizer.from_pretrained("datamol-io/safe-gpt")

    def embed(self, smiles_list):
        safe_strings =  [encode(smiles, slicer="mmpa") for smiles in smiles_list]
        encoded = self.tokenizer.encode(safe_strings, ids_only=True)

        max_seq_length = max(len(seq) for seq in encoded) 
        padded = [seq + [self.tokenizer.pad_token_id] * (max_seq_length - len(seq)) for seq in encoded]
        tokens_tensor = torch.tensor(padded)
        return tokens_tensor


if __name__ == "__main__":
    smiles_list = ["CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2)O", "CN(C)CC=CC(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OC4CCOC4", "CNC(=O)C1=CC=CC=C1SC2=CC3=C(C=C2)C(=NN3)C=CC4=CC=CC=N4"]
    safe_embeddings = SafeEmbeddingGPT().embed(smiles_list)
    print(f"SAFE-GPT Embeddings Shape: {safe_embeddings.shape}")
    print(f"SAFE-GPT Embeddings Tensor:\n{safe_embeddings}\n")
