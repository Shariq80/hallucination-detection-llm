import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path

from src.data.wiki_pages import fetch_wiki_pages, save_pages
from src.retrieval.preprocessing import clean_text, chunk_text

class IndexBuilder:

    def __init__(self, config):
        model_name = config.get("models", "embedding_model")
        self.model = SentenceTransformer(model_name)

        self.batch_size = config.get("retrieval", "embedding_batch_size")
        self.sentences_per_chunk = config.get("retrieval", "sentences_per_chunk")

        self.index_path = Path(config.get("retrieval", "index_path"))
        self.metadata_path = Path(config.get("retrieval", "metadata_path"))

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def build(self, wiki_pages):

        chunks = []
        metadata = []

        for page in wiki_pages:

            clean = clean_text(page["text"])

            page_chunks = chunk_text(clean, sentences_per_chunk=self.sentences_per_chunk)

            for chunk in page_chunks:

                chunks.append(page["title"] + ". " + chunk)

                metadata.append({
                    "title": page["title"],
                    "text": chunk
                })

        embeddings = self.model.encode(
            chunks,
            batch_size=self.batch_size,
            show_progress_bar=True
        )

        embeddings = np.array(embeddings).astype("float32")

        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        return index, metadata


    def save(self, index, metadata):
        faiss.write_index(index, str(self.index_path))

        with open(self.metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        
        print(f"Index saved to {self.index_path}")
        print(f"Metadata saved to {self.metadata_path}")