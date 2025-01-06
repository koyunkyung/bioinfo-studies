import pandas as pd
from sklearn.preprocessing import LabelEncoder
import requests

# Data preprocessing
def preprocess(input_file, output_file):
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

    # Group by 'disease' and aggregate
    grouped_data = merged_data.groupby('disease').agg({
        'cell_line_name': lambda x: ', '.join(x),  # Combine cell line names
        'drug_name': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',  # Get most frequent drug name
        'putative_target': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',  # Get most frequent target
        'ln_ic50': 'mean'  # Calculate mean for ln_ic50
    }).reset_index()

    ## Encoding and Embedding ##
    # Encode categorical variables
    cell_line_encoder = LabelEncoder()
    drug_encoder = LabelEncoder()
    filtered_data['cell_line_encoded'] = cell_line_encoder.fit_transform(filtered_data['cell_line_name'])
    filtered_data['drug_encoded'] = drug_encoder.fit_transform(filtered_data['drug_name'])

    # Save the processed data and encoders
    filtered_data.to_csv("data/GDSC2_processed.csv", index=False)
    cell_line_encoder.classes_.tofile("data/cell_line_classes.txt", sep="\n")
    drug_encoder.classes_.tofile("data/drug_classes.txt", sep="\n")
    print(f"Processed data saved to {output_file}")



if __name__ == "__main__":
    preprocess("data/GDSC2_raw.xlsx", "data/GDSC2_processed.csv")