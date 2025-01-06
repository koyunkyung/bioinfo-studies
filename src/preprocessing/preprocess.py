from src.preprocessing.data_cleaning import clean_data
from src.preprocessing.embeddings import scbert_embedding

def preprocess_pipeline(raw_file, cleaned_file, processed_file):
    """
    Run the full preprocessing pipeline.
    """
    # Step 1: Clean the raw dataset
    clean_data(raw_file, cleaned_file)

    # Step 2: Generate SCBERT embeddings
    scbert_embedding(cleaned_file, processed_file)

if __name__ == "__main__":
    preprocess_pipeline(
        raw_file="data/raw/GDSC2_raw.xlsx",
        cleaned_file="data/processed/GDSC2_cleaned.csv",
        processed_file="data/processed/GDSC2_processed.csv"
    )
