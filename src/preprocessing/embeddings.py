import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch
import numpy as np


### 1. LabelEncoder 기반 임베딩 ###
def label_encoding(data, column_name):
    label_encoder = LabelEncoder()
    data[f"{column_name}_label_encoded"] = label_encoder.fit_transform(data[column_name])
    print(f"LabelEncoder embeddings added for column: {column_name}")
    return data


### 2. SCBERT 기반 임베딩 ###
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


### 3. RDKit 분자 특성 기반 임베딩 ###
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