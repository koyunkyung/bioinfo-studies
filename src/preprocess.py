import pandas as pd
data = pd.read_excel("data/GDSC2_raw.xlsx", sheet_name=0)
data.columns = data.columns.str.lower()
#print(data.info())

# Select relevant variables to use for the analysis
columns_to_extract = ['cosmic_id', 'cell_line_name', 'drug_id', 'drug_name', 'putative_target', 'ln_ic50', 'auc', 'z_score']
selected_data = data[columns_to_extract]

# Filter outliers and NA values
filtered_data = selected_data[
    (data['auc'] >= 0.2) & (data['auc'] <= 0.8) &
    (data['z_score'] >= -2) & (data['z_score'] <= 2)
]
filtered_data = filtered_data.drop(columns=['auc', 'z_score'])
filtered_data = filtered_data.dropna()
#print(filtered_data.isnull().sum())

# Save the processed dataset
filtered_data.to_csv("data/GDSC2_processed.csv", index=False)