import pandas as pd
import requests
from fetch_smiles import fetch_smiles_with_cache
from sklearn.preprocessing import StandardScaler

def fetch_smiles_wrapper(drug_name, session):
    return fetch_smiles_with_cache(drug_name, session)

def clean_data(input_file, output_file):

    data = pd.read_excel(input_file, sheet_name=0)
    data.columns = data.columns.str.lower()

    annotations = pd.read_csv("data/raw/cell_line_annotations.txt", sep="\t")
    annotations = annotations[['Name', 'Disease']]
    annotations.columns = ['cell_line_name', 'disease']

    columns_to_extract = ['cell_line_name', 'drug_name', 'putative_target', 'ln_ic50', 'auc', 'z_score']
    selected_data = data[columns_to_extract]

    # 필터링 기준: z-score, auc, (ln_ic50 > 0)
    filtered_data = selected_data[
        ((selected_data['z_score'] <= -2) | (selected_data['z_score'] >= 2)) &
        ((selected_data['auc'] <= 0.35) | (selected_data['auc'] >= 0.85)) &
        (selected_data['ln_ic50'] > 0)
    ]
    filtered_data = filtered_data.drop(columns=['auc', 'z_score']).dropna()

    # disease 정보 합치기
    merged_data = pd.merge(filtered_data, annotations, on='cell_line_name', how='left')
    merged_data = merged_data.dropna(subset=['disease'])

    # SMILES 데이터 합치기
    with requests.Session() as session:
        merged_data['smiles'] = merged_data['drug_name'].apply(lambda x: fetch_smiles_wrapper(x, session))
    merged_data = merged_data.dropna(subset=['smiles'])

    # 정규화
    scaler = StandardScaler()
    merged_data['ln_ic50'] = scaler.fit_transform(merged_data[['ln_ic50']])

    merged_data.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    clean_data(
        input_file="data/raw/GDSC2_raw.xlsx",
        output_file="data/processed/GDSC2_cleaned.csv"
    )