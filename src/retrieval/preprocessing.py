import re
from nltk.tokenize import sent_tokenize


def clean_text(text):
    text = re.sub(r'\[[0-9]*\]', '', text) # removes citations
    text = re.sub(r'\n', ' ', text) # removes line breaks
    text = re.sub(r'\s+', ' ', text) # removes extra spaces

    return text

def chunk_text(text, sentences_per_chunk):
    sentences = sent_tokenize(text)
    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i+sentences_per_chunk])

        if len(chunk) > 40:
            chunks.append(chunk)

    return chunks