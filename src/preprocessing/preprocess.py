import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
import torch

def preprocess(input_file, cleaned_file, processed_file):
    """
    Filter and encode the raw dataset.
    """
    # Load main dataset
    data = pd.read_excel(input_file, sheet_name=0)
    data.columns = data.columns.str.lower()

    # Load cell_line_annotations.txt and create mapping
    annotations = pd.read_csv("data/raw/cell_line_annotations.txt", sep="\t")
    annotations = annotations[['Name', 'Disease']]
    annotations.columns = ['cell_line_name', 'disease']

    ## Filtering ##
    # Select relevant variables to use for the analysis
    columns_to_extract = ['cell_line_name', 'drug_name', 'putative_target', 'ln_ic50', 'auc', 'z_score']
    selected_data = data[columns_to_extract]

    # Filter out only data that are resistant or sentitive to drugs
    filtered_data = selected_data[
        (data['z_score'] <= -2) | (data['z_score'] >= 2) &
        (data['auc'] <= 0.35) | (data['auc'] >= 0.85)
    ]
    # Drop out unnecessary columns and NA values
    filtered_data = filtered_data.drop(columns=['auc', 'z_score'])
    filtered_data = filtered_data.dropna()

    # Merge with annotations to add disease information
    merged_data = pd.merge(filtered_data, annotations, on='cell_line_name', how='left')
    merged_data.to_csv(cleaned_file, index=False)
    print(f"Cleaned data saved to {cleaned_file}")

    merged_data['smiles'] = merged_data['drug_name'].apply(fetch_smiles)

    # Group by 'disease' and aggregate
    grouped_data = merged_data.groupby('disease').agg({
        'cell_line_name': lambda x: ', '.join(x),  # Combine cell line names
        'drug_name': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',  # Get most frequent drug name
        'putative_target': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',  # Get most frequent target
        'ln_ic50': 'mean',  # Calculate mean for ln_ic50
        'smiles': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'  # Get most frequent SMILES
    }).reset_index()

    # Create input sequence
    input_sequences = [
        f"cell_line: {row['cell_line_name']} [SEP] drug_name: {row['drug_name']} [SEP] putative_target: {row['putative_target']} [SEP] SMILES: {row['smiles']} [SEP] disease: {row['disease']}"
        for _, row in grouped_data.iterrows()
    ]
    
    ## Tokenization and embedding using scBERT ##
    tokenizer = AutoTokenizer.from_pretrained("havens2/scBERT_SER")
    encoded_inputs = tokenizer(
        input_sequences,
        padding=True,
        truncation=True,
        return_tensors="pt"  # PyTorch 텐서로 반환
    )
    # Load scBERT model
    model = AutoModel.from_pretrained("havens2/scBERT_SER")
    with torch.no_grad():
        outputs = model(**encoded_inputs)

    # Extract CLS token embeddings
    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embedding

    # Save embeddings for future use
    torch.save(cls_embeddings, "data/drug_smiles_target_embeddings.pt")
    print("CLS embeddings saved at 'data/drug_smiles_target_embeddings.pt'")

    # Save the grouped data for reference
    grouped_data.to_csv(processed_file, index=False)
    print(f"Grouped data saved to {processed_file}")


if __name__ == "__main__":
    preprocess(
        input_file="data/raw/GDSC2_raw.xlsx",
        cleaned_file="data/processed/GDSC2_cleaned.csv",
        processed_file="data/processed/GDSC2_processed.csv"
    )


from preprocessing.data_cleaning import clean_data
from preprocessing.embeddings import generate_scbert_embeddings

def preprocess_pipeline(raw_file, cleaned_file, processed_file):
    """
    Run the full preprocessing pipeline.
    """
    # Step 1: Clean the raw dataset
    clean_data(raw_file, cleaned_file)

    # Step 2: Generate SCBERT embeddings
    generate_scbert_embeddings(cleaned_file, processed_file)

if __name__ == "__main__":
    preprocess_pipeline(
        raw_file="data/raw/GDSC2_raw.xlsx",
        cleaned_file="data/processed/GDSC2_cleaned.csv",
        processed_file="data/processed/GDSC2_processed.csv"
    )
