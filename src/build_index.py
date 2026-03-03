# src/build_index.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from preprocessing import build_chunks


def build_faiss_index(chunks, model_name="all-MiniLM-L6-v2", batch_size=64):
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings


def save_index(index, chunks, path="index"):
    faiss.write_index(index, f"{path}.faiss")

    with open(f"{path}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


if __name__ == "__main__":
    # Replace with your wiki_pages loading logic
    wiki_pages = ...

    chunks = build_chunks(wiki_pages)
    index, embeddings = build_faiss_index(chunks)

    save_index(index, chunks)