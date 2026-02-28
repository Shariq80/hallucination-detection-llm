import pandas as pd
import requests
import re
import os

URLS = {
    "train": "https://fever.ai/download/fever/train.jsonl",
    "dev": "https://fever.ai/download/fever/shared_task_dev.jsonl"
}

def clean_text(text):
    """Clean and normalize the text."""
    if not isinstance(text, str):
        return ""
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def download_file(url, target_path):
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded to {target_path}")

def process_fever():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    
    for split, url in URLS.items():
        print(f"\n--- Processing {split} split ---")
        raw_path = f"data/raw/fever_{split}.jsonl"
        
        if not os.path.exists(raw_path):
            download_file(url, raw_path)
        else:
            print(f"File {raw_path} already exists. Skipping download.")
            
        print("Loading dataset...")
        df = pd.read_json(raw_path, lines=True)
        
        print("Cleaning and preprocessing claims...")
        print(f"Original size: {len(df)}")
        
        # Drop rows where claim is missing
        if 'claim' in df.columns:
            df = df.dropna(subset=['claim'])
            # Clean the text of the claim
            df['claim_clean'] = df['claim'].apply(clean_text)
        else:
            print("Warning: 'claim' column not found.")
        
        print("Converting dataset into useable format...")
        output_path = f"data/processed/fever_{split}.jsonl"
        df.to_json(output_path, orient="records", lines=True)
        print(f"Saved processed {split} dataset to {output_path} (Size: {len(df)})")

if __name__ == "__main__":
    process_fever()
