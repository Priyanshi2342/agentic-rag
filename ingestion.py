import os
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 🔥 Embedding model (best for RAG tasks)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


# -----------------------------
# 1. LOAD DOCUMENTS
# -----------------------------
def load_documents(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                documents.append({
                    "text": f.read(),
                    "source": file
                })

    return documents


# -----------------------------
# 2. STRUCTURE-AWARE SPLITTING
# -----------------------------
def split_by_structure(text):
    """
    Split text using headings, separators, and paragraph breaks
    """
    sections = re.split(r'\n\s*\n|---|SECTION \d+|ISSUE \d+', text)
    return [s.strip() for s in sections if len(s.strip()) > 50]


# -----------------------------
# 3. CHUNKING WITH OVERLAP
# -----------------------------
def chunk_text(text, chunk_size=200, overlap=40):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 30:  # avoid tiny chunks
            chunks.append(chunk)

    return chunks


# -----------------------------
# 4. PROCESS DOCUMENTS
# -----------------------------
def process_documents(folder_path):
    docs = load_documents(folder_path)

    all_chunks = []
    metadata = []

    for doc in docs:
        sections = split_by_structure(doc["text"])

        for section in sections:
            chunks = chunk_text(section)

            for chunk in chunks:
                all_chunks.append(chunk)
                metadata.append({
                    "source": doc["source"],
                    "preview": chunk[:120]
                })

    # -----------------------------
    # 5. EMBEDDINGS
    # -----------------------------
    embeddings = model.encode(
        all_chunks,
        normalize_embeddings=True  # important for cosine similarity
    )

    embeddings = np.array(embeddings).astype("float32")

    # -----------------------------
    # 6. FAISS INDEX (COSINE SIM)
    # -----------------------------
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, all_chunks, metadata


# -----------------------------
# 7. SAVE INDEX
# -----------------------------
def save_index(index, chunks, metadata):
    faiss.write_index(index, "faiss_index.bin")
    np.save("chunks.npy", np.array(chunks, dtype=object))
    np.save("metadata.npy", np.array(metadata, dtype=object))


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    index, chunks, metadata = process_documents("data")
    save_index(index, chunks, metadata)

    print(f"✅ Indexed {len(chunks)} chunks successfully!")