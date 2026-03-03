# src/preprocessing.py

def clean_titles(page_titles):
    return [title.replace("_", " ") for title in page_titles if title is not None]


def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def build_chunks(wiki_pages):
    all_chunks = []

    for page in wiki_pages:
        title = page["title"]
        text = page["text"]

        chunks = chunk_text(text)

        for chunk in chunks:
            full_text = f"{title}. {chunk}"
            all_chunks.append(full_text)

    return all_chunks