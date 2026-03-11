import pandas as pd
import requests
import re
import os
import json

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

def extract_titles(df):
    titles = set()
    
    if "evidence" not in df.columns:
        return []
    
    for evidence_groups in df["evidence"]:
        for group in evidence_groups:
            for item in group:
                if len(item)>2 and item[2]:
                    titles.add(item[2].replace("_", " "))
    return list(titles)


def process_fever():

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)

    all_titles = set()

    for split, url in URLS.items():

        print(f"\n--- Processing {split} split ---")

        raw_path = f"data/raw/fever_{split}.jsonl"

        if not os.path.exists(raw_path):
            download_file(url, raw_path)

        print("Loading dataset...")

        df = pd.read_json(raw_path, lines=True)

        df = df.dropna(subset=["claim"])

        df["claim_clean"] = df["claim"].apply(clean_text)

        titles = extract_titles(df)

        all_titles.update(titles)

        output_path = f"data/processed/fever_{split}.jsonl"

        df.to_json(output_path, orient="records", lines=True)

        print(f"Saved processed {split} dataset")

    titles_path = "data/processed/wiki_titles.json"

    with open(titles_path, "w") as f:
        json.dump(list(all_titles), f)

    print(f"Saved {len(all_titles)} Wikipedia titles")

if __name__ == "__main__":
    process_fever()
