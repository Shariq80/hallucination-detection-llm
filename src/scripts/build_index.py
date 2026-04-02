import json
import argparse
from pathlib import Path

from src.utils.config import Config
from src.data.wiki_pages import fetch_wiki_pages, save_pages
from src.retrieval.build_index import IndexBuilder


def main():
    # -----------------------------
    # Parse arguments
    # -----------------------------
    parser = argparse.ArgumentParser(description="Build FAISS index for retrieval")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag to differentiate outputs (e.g., test, debug)"
    )
    args = parser.parse_args()

    # -----------------------------
    # Load config
    # -----------------------------
    print(f"\nLoading config: {args.config}")
    config = Config(args.config)

    titles_path = "data/processed/wiki_titles.json"
    limit = config["data"]["wiki_fetch_limit"]

    if not Path(titles_path).exists():
        raise FileNotFoundError("Run prepare_fever.py first")

    with open(titles_path) as f:
        titles = json.load(f)

    print(f"Found {len(titles)} wiki titles")
    print(f"Fetch limit: {limit}")

    # -----------------------------
    # Fetch Wikipedia pages
    # -----------------------------
    print("\nFetching Wikipedia pages...")
    wiki_pages = fetch_wiki_pages(titles, limit=limit)

    print(f"Fetched {len(wiki_pages)} pages")

    # -----------------------------
    # Save raw pages (optional separation)
    # -----------------------------
    if args.tag:
        print(f"Saving pages with tag: {args.tag}")
        save_pages(wiki_pages, suffix=args.tag)  # <-- requires small change in save_pages
    else:
        print("Saving pages (default)")
        save_pages(wiki_pages)

    # -----------------------------
    # Build FAISS index
    # -----------------------------
    print("\nBuilding FAISS index...")

    builder = IndexBuilder(config)

    index, metadata = builder.build(wiki_pages)

    # -----------------------------
    # Save index
    # -----------------------------
    index_path = config["retrieval"]["index_path"]
    metadata_path = config["retrieval"]["metadata_path"]

    print(f"\nSaving index to: {index_path}")
    print(f"Saving metadata to: {metadata_path}")

    builder.save(index, metadata)

    print("\nIndex build complete!")


if __name__ == "__main__":
    main()