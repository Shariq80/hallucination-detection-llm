import os
import pickle
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore

# 1. Paths to your specific files
index_file = r"E:/GitHub - repos/hallucination-detection-llm/indexes/faiss_2_sentences.index"
pkl_file = r"E:/GitHub - repos/hallucination-detection-llm/indexes/metadata_2_sentences.pkl"

# 2. Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

try:
    # 3. Manually read the FAISS index file
    index_data = faiss.read_index(index_file)

    # 4. Manually load the metadata/documents list
    with open(pkl_file, "rb") as f:
        doc_list = pickle.load(f)

    # 5. Handle the case where the pickle is a LIST of documents
    if isinstance(doc_list, list):
        # Create unique IDs for each document (e.g., "0", "1", "2"...)
        doc_dict = {str(i): doc for i, doc in enumerate(doc_list)}
        index_to_docstore_id = {i: str(i) for i in range(len(doc_list))}
        docstore = InMemoryDocstore(doc_dict)
    else:
        # Fallback for standard LangChain tuple format
        docstore, index_to_docstore_id = doc_list

    # 6. Reconstruct the FAISS object
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index_data,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    # 7. Print the data
    print(f"Successfully loaded {len(vectorstore.docstore._dict)} documents.")
    for doc_id, doc in vectorstore.docstore._dict.items():
        print(f"\n--- Doc: {doc_id} ---")
        # Handle cases where 'doc' might be a dict or a LangChain Document object
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        print(f"Content: {content[:200]}...")

except Exception as e:
    print(f"Failed to load: {e}")
