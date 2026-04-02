import wikipedia
import json
import time
from wikipedia.exceptions import DisambiguationError, PageError
from pathlib import Path


def fetch_wiki_pages(titles, delay=0.2, limit=None):
    """
    Fetch Wikipedia pages for given titles.
    """

    pages = []
    total = len(titles)

    if limit:
        titles = titles[:limit]
    total = len(titles)

    print(f"Fetching {total} Wikipedia pages...\n")

    for i, title in enumerate(titles):

        if i % 50 == 0:
            print(f"Processed {i}/{total}")

        try:
            page = wikipedia.page(title, auto_suggest=False)

            pages.append({
                "title": page.title,
                "text": page.content
            })

        except DisambiguationError as e:

            try:
                page = wikipedia.page(e.options[0])

                pages.append({
                    "title": page.title,
                    "text": page.content
                })

            except Exception:
                continue

        except PageError:
            continue

        except Exception as e:
            print(f"Skipping {title}: {e}")
            continue

        # polite delay to avoid rate limits
        time.sleep(delay)

    print(f"\nFetched {len(pages)} pages successfully")

    return pages


def save_pages(pages, suffix=None):
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    if suffix:
        file_path = f"data/processed/wiki_pages_{suffix}.json"
    else:
        file_path = "data/processed/wiki_pages.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print(f"Saved pages to {file_path}")


def load_pages(path="data/processed/wiki_pages.json"):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    with open(path, encoding="utf-8") as f:
        pages = json.load(f)

    print(f"Loaded {len(pages)} pages")

    return pages