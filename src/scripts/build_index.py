import json
from pathlib import Path

from src.utils.config import Config
from src.data.wiki_pages import fetch_wiki_pages, save_pages
from src.retrieval.build_index import IndexBuilder


def main():

    print("Loading config...")

    config = Config("configs/default.yaml")

    titles_path = "data/processed/wiki_titles.json"
    limit = config["data"]["wiki_fetch_limit"]

    if not Path(titles_path).exists():
        raise FileNotFoundError("Run prepare_fever.py first")

    with open(titles_path) as f:
        titles = json.load(f)

    print(f"Found {len(titles)} wiki titles")

    print("Fetching Wikipedia pages...")
    wiki_pages = fetch_wiki_pages(titles, limit=limit)

    save_pages(wiki_pages)

    print("Building FAISS index...")

    builder = IndexBuilder(config)

    index, metadata = builder.build(wiki_pages)

    builder.save(index, metadata)

    print("Index build complete")


if __name__ == "__main__":
    main()