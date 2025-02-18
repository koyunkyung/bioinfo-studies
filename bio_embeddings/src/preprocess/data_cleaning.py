import pandas as pd
import requests
from fetch_smiles import fetch_smiles_with_cache
from sklearn.preprocessing import StandardScaler
import torch

# drug_name와 매칭되는 SMILES 데이터 가져오는 함수
def fetch_smiles_wrapper(drug_name, session):
    return fetch_smiles_with_cache(drug_name, session)

# 메인으로 데이터 전처리 해주는 함수
def clean_data(input_file, output_file):

    data = pd.read_excel(input_file, sheet_name=0)
    data.columns = data.columns.str.lower()

    annotations = pd.read_csv("data/raw/cell_line_annotations.txt", sep="\t")
    annotations = annotations[['Name', 'Disease']]
    annotations.columns = ['cell_line_name', 'disease']

    columns_to_extract = ['cell_line_name', 'drug_name', 'putative_target', 'ln_ic50', 'auc', 'z_score']
    selected_data = data[columns_to_extract]

    # 필터링 기준: z-score, auc
    selected_data.loc[:, 'z_score'] = pd.to_numeric(selected_data['z_score'])
    selected_data.loc[:, 'auc'] = pd.to_numeric(selected_data['auc'])

    filtered_data = selected_data[
        ((selected_data['z_score'] <= -2) | (selected_data['z_score'] >= 2)) &
        ((selected_data['auc'] <= 0.35) | (selected_data['auc'] >= 0.85)) &
        (selected_data['ln_ic50'] > 0)
    ]
    filtered_data = filtered_data.drop(columns=['auc', 'z_score']).dropna()

    # disease 정보 합치고 열 하나로 만들기
    merged_data = pd.merge(filtered_data, annotations, on='cell_line_name', how='left')
    merged_data = merged_data.dropna(subset=['disease'])
    cell_cols = ['cell_line_name', 'disease']
    merged_data['combined_cell_line'] = merged_data[cell_cols].apply(lambda row: ':'.join(row.values.astype(str)), axis=1)
    merged_data = merged_data.drop(columns=['cell_line_name', 'disease'])

    # SMILES 데이터 합치기
    with requests.Session() as session:
        merged_data['drug_smiles'] = merged_data['drug_name'].apply(lambda x: fetch_smiles_wrapper(x, session))
    merged_data = merged_data.dropna(subset=['drug_smiles'])

    # drug_name이랑 putative_target 열 하나로 만들기
    drug_cols = ['drug_name', 'putative_target']
    merged_data['combined_drug'] = merged_data[drug_cols].apply(lambda row: ':'.join(row.values.astype(str)), axis=1)
    merged_data = merged_data.drop(columns=['drug_name', 'putative_target'])

    # ic50 점수 (수치형 데이터) 정규화
    scaler = StandardScaler()
    merged_data['ln_ic50'] = scaler.fit_transform(merged_data[['ln_ic50']])
    merged_data = merged_data[['combined_cell_line', 'combined_drug', 'drug_smiles', 'ln_ic50']]

    merged_data.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")


if __name__ == "__main__":
    clean_data(
        input_file="data/raw/GDSC2_raw.xlsx",
        output_file="data/processed/GDSC2_cleaned.csv"
    )