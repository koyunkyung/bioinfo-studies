import torch
import json
from transformers import GPT2Config, GPT2LMHeadModel
from scGPT_human.gene_tokenizer import get_default_gene_vocab, tokenize_and_pad_batch

class scGPTEmbedding:
    def __init__(self, model_dir):
        model_path = f"{model_dir}/best_model.pt"
        args_path = f"{model_dir}/args.json"
        with open(args_path, 'r') as f:
            self.args = json.load(f)

        self.vocab = get_default_gene_vocab()

        config = GPT2Config(
            vocab_size=len(self.vocab),
            n_embd=self.args.get("embsize", 512),
            n_layer=self.args.get("nlayers", 12),
            n_head=self.args.get("nheads", 8),
            n_inner=self.args.get("d_hid", 512),
        )
        self.model = GPT2LMHeadModel(config)
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def embed(self, cell_lines, max_len=512):
        # Cell line 데이터를 ID로 변환
        tokenized_data = [
            [self.vocab[token] for token in line.split(":")] for line in cell_lines
        ]

        padded_batch = tokenize_and_pad_batch(
            data=tokenized_data,
            gene_ids=list(range(len(self.vocab))),
            max_len=max_len,
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=0
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=padded_batch["genes"],
                attention_mask=(padded_batch["genes"] != self.vocab["<pad>"]),
                output_hidden_states=True
            )

        embeddings = outputs.hidden_states[-1][:, 0, :]
        return embeddings 


if __name__ == "__main__":

    combined_cell_line = ["Camptothecin:TOP1", "Vinblastine:Microtubule destabiliser", "Cisplatin:DNA crosslinker"]
    smiles_list = ["CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2)O", "CN(C)CC=CC(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OC4CCOC4", "CNC(=O)C1=CC=CC=C1SC2=CC3=C(C=C2)C(=NN3)C=CC4=CC=CC=N4"]
    model_dir = "scGPT_human"

    # scGPT를 사용한 cell line 임베딩
    scgpt = scGPTEmbedding(model_dir)
    cell_embeddings = scgpt.embed(combined_cell_line)
    print(f"scGPT Cell Embeddings Shape: {cell_embeddings.shape}")
    print(f"scGPT Cell Embeddings Tensor:\n{cell_embeddings}")
