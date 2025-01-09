import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec

### Cell line name과 disease 관련 임베딩 클래스 ###
class CellLineEmbedding:
    def __init__(self, vector_size=100, window=5, min_count=1, epochs=100):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs

    def train_word2vec(self, data):
        """
        param data: List of sentences, where each sentence is a list of words (e.g., [["cell_line1", "disease1"], ...])
        return: Trained Word2Vec model (gensim).
        """
        model = Word2Vec(sentences=data, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=4, epochs=self.epochs)
        return model
    
    def get_embedding(self, model, word):
        """
        retrieve the embedding for a specific word from the Word2Vec model
        param model: Trained Word2Vec model.
        param word: Word to retrieve the embedding for.
        return: Numpy array containing the embedding vector.
        """
        if word in model.wv:
            return model.wv[word]
        else:
            print(f"Word '{word}' not in vocabulary!")
            return None

    
### Drug 관련 임베딩 클래스 ###
class DrugEmbedding:
    def _init__(self):
        pass

    # 1. One-Hot Encoding
    def one_hot_encoding(self, drug_names):
        """
        param drug_names: List of drug names (e.g., ["DrugA", "DrugB"])
        return: One-hot encoded numpy array.
        """
        encoder = OneHotEncoder()
        encoded = encoder.fit_transform(np.array(drug_names).reshape(-1, 1))
        return encoded
    
    # 2. Pre-trained scBERT Embedding: transformer model designed for single-cell gene expression data
    def scbert_embedding(self, smiles_list):
        """
        param smiles_list: List of SMILES strings.
        return: PyTorch tensor containing the embeddings.
        """
        tokenizer = AutoTokenizer.from_pretrained("havens2/scBERT_SER")
        model = AutoModel.from_pretrained("havens2/scBERT_SER")

        # tokenize SMILES strings
        inputs = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # use the mean pooling of token embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings
    
    # 3. Pre-trained ChemBERTa Embedding: trained on SMILES strings that represent chemical molecules
    def chemberta_embedding(self, smiles_list):
        """
        param smiles_list: List of SMILES strings.
        return: PyTorch tensor containing the embeddings.
        """
        tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

        # tokenize SMILES strings
        inputs = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # use the mean pooling of token embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings
    

### Example Usage ###
if __name__ == "__main__":

    # cell line embedding examples
    cell_line_data = [
        ["cell_line1", "disease1"],
        ["cell_line2", "disease2"],
        ["cell_line3", "disease3"]
    ]

    cell_line_embedding = CellLineEmbedding(vector_size=50, window=3, min_count=1, epochs=5)
    word2vec_model = cell_line_embedding.train_word2vec(cell_line_data)
    
    print("Embedding for 'cell_line1':", cell_line_embedding.get_embedding(word2vec_model, "cell_line1"))
    print("Embedding for 'disease1':", cell_line_embedding.get_embedding(word2vec_model, "disease1"))

    # drug embedding examples
    drug_embedding = DrugEmbedding()
    drug_names = ["DrugA", "DrugB", "DrugC"]
    smiles_list = ["CCO", "CCN", "CCC"]

    one_hot = drug_embedding.one_hot_encoding(drug_names)
    print("One-Hot Encoding:", one_hot)

    scbert_embeddings = drug_embedding.scbert_embedding(smiles_list)
    print("scBERT Embeddings:", scbert_embeddings)
    
    chemberta_embeddings = drug_embedding.chemberta_embedding(smiles_list)
    print("ChemBERTa Embeddings:", chemberta_embeddings)


