import os
import json
import requests
from multiprocessing import Pool
import pandas as pd
from preprocessing.fetch_smiles import fetch_smiles_parallel

def clean_data(input_file, output_file):
    """
    Clean and preprocess the raw dataset.
    """
    # Load main dataset
    data = pd.read_excel(input_file, sheet_name=0)
    data.columns = data.columns.str.lower()

    # Load cell_line_annotations.txt and create mapping
    annotations = pd.read_csv("data/raw/cell_line_annotations.txt", sep="\t")
    annotations = annotations[['Name', 'Disease']]
    annotations.columns = ['cell_line_name', 'disease']

    # Select relevant variables to use for the analysis
    columns_to_extract = ['cell_line_name', 'drug_name', 'putative_target', 'ln_ic50', 'auc', 'z_score']
    selected_data = data[columns_to_extract]

    # Filter and drop unnecessary data
    filtered_data = selected_data[
        ((selected_data['z_score'] <= -2) | (selected_data['z_score'] >= 2)) &
        ((selected_data['auc'] <= 0.35) | (selected_data['auc'] >= 0.85))
    ]
    filtered_data = filtered_data.drop(columns=['auc', 'z_score']).dropna()

    # Merge with annotations to add disease information
    merged_data = pd.merge(filtered_data, annotations, on='cell_line_name', how='left')
    merged_data['smiles'] = merged_data['drug_name'].apply(fetch_smiles_parallel)

    # Group by 'disease' and aggregate
    grouped_data = merged_data.groupby('disease').agg({
        'cell_line_name': lambda x: ', '.join(x),  # Combine cell line names
        'drug_name': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',  # Get most frequent drug name
        'putative_target': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',  # Get most frequent target
        'ln_ic50': 'mean',  # Calculate mean for ln_ic50
        'smiles': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'  # Get most frequent SMILES
    }).reset_index()

    # Save the cleaned dataset
    grouped_data.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    clean_data(
        input_file="data/raw/GDSC2_raw.xlsx",
        output_file="data/processed/GDSC2_cleaned.csv"
    )