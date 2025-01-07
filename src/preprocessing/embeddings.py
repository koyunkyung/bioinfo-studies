import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import KeyedVectors
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch
import numpy as np


### 1. LabelEncoder 기반 임베딩 ###
def label_encoding(data, column_name):
    label_encoder = LabelEncoder()
    data[f"{column_name}_encoded"] = label_encoder.fit_transform(data[column_name])
    print(f"LabelEncoder embeddings added for column: {column_name}")
    return data

# ### 2. Word Frequency 기반 임베딩 ###
# def word_freq_embedding(data, column_name):
#      vectorizer = CountVectorizer()
#      freq_vectors = vectorizer.fit_transform(data[column_name]).toarray()
#      feature_names = vectorizer.get_feature_names_out()
#      print(f"Word Frequency embeddings generated for column: {column_name}. Shape: {freq_vectors.shape}")

#      # add word frequency embeddings as columns
#      freq_df = pd.DataFrame(freq_vectors, columns=[f"{column_name}_word_freq_{name}" for name in feature_names])
#      data = pd.concat([data.reset_index(drop=True), freq_df], axis=1)

#      return data

# ### 3. Gene2Vec 기반 임베딩 ###
# def gene2vec_embedding(data, column_name, model_path):
#      print(f"Loading Gene2Vec model from {model_path}...")
#      gene2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

#      gene2vec_vectors = []
#      for name in data[column_name]:
#         if name in gene2vec_model:
#                gene2vec_vectors.append(gene2vec_model[name])
#         else:
#              gene2vec_vectors.append(np.zeros(gene2vec_model.vector_size))

#      gene2vec_df = pd.DataFrame(gene2vec_vectors, columns=[f"{column_name}_gene2vec_{i}" for i in range(gene2vec_model.vector_size)])
#      data = pd.concat([data.reset_index(drop=True), gene2vec_df], axis=1)

#      print(f"Gene2Vec embeddings added for column: {column_name}. Shape: {gene2vec_df.shape}")
#      return data
               

### 4. SCBERT 기반 임베딩 ###
def scbert_embedding(data, model_name="havens2/scBERT_SER"):
    # Create input sequences
    input_sequences = [
        f"cell_line: {row['cell_line_name']} [SEP] drug_name: {row['drug_name']} [SEP] putative_target: {row['putative_target']} [SEP] SMILES: {row['smiles']} [SEP] disease: {row['disease']}"
        for _, row in data.iterrows()
    ]

    # Tokenize and encode
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded_inputs = tokenizer(input_sequences, padding=True, truncation=True, return_tensors="pt")

    # Load SCBERT model
    model = AutoModel.from_pretrained(model_name)
    with torch.no_grad():
        outputs = model(**encoded_inputs)

    # Extract CLS token embeddings
    cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()

    # Add embeddings to DataFrame
    data['scbert_embedding'] = list(cls_embeddings)
    print("SCBERT embeddings generated and added to the DataFrame.")
    return data


### 5. RDKit 분자 특성 기반 임베딩 ###
def rdkit_molecular(data, smiles_column):
    def compute_molecular(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
                return [
                    Descriptors.MolWt(mol),        # Molecular weight
                    Descriptors.MolLogP(mol),     # LogP
                    Descriptors.NumHDonors(mol),  # Number of H-bond donors
                    Descriptors.NumHAcceptors(mol)  # Number of H-bond acceptors
                ]
        else:
            return [np.nan, np.nan, np.nan, np.nan]

    # Compute molecular features for each SMILES
    molecular_features = np.array([compute_molecular(smiles) for smiles in data[smiles_column]])
    feature_columns = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors']
    molecular_features_df = pd.DataFrame(molecular_features, columns=feature_columns)

    # Concatenate with original DataFrame
    data = pd.concat([data.reset_index(drop=True), molecular_features_df], axis=1)
    print("RDKit molecular features generated and added to the DataFrame.")
    return data
 

### 통합 함수 ###
def generate_all_embeddings(data, smiles_column="smiles"):
    # 1. LabelEncoder embeddings
    data = label_encoding(data, "cell_line_name")
    data = label_encoding(data, "drug_name")

    # 2. SCBERT embeddings
    data = scbert_embedding(data)

    # 3. RDKit molecular features
    data = rdkit_molecular(data, smiles_column)

    return data

### 임베딩 데이터 저장 ###
data = pd.read_csv("data/processed/GDSC2_cleaned.csv")
data_with_embeddings = generate_all_embeddings(data, smiles_column="smiles")
data_with_embeddings.to_csv("data/processed/GDSC2_embeddings.csv", index=False)
print("All embeddings have been generated and saved to 'data/processed/GDSC2_embeddings.csv'.")