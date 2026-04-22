import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# same model as ingestion
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


# -----------------------------
# LOAD DATA
# -----------------------------
def load_index():
    index = faiss.read_index("faiss_index.bin")
    chunks = np.load("chunks.npy", allow_pickle=True)
    metadata = np.load("metadata.npy", allow_pickle=True)
    return index, chunks, metadata


# -----------------------------
# RETRIEVE TOP-K
# -----------------------------
def retrieve(query, top_k=5):
    index, chunks, metadata = load_index()

    # encode query
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    # search
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "chunk": chunks[idx],
            "score": float(scores[0][i]),
            "source": metadata[idx]["source"]
        })

    return results


# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    query = "What are high-risk AI systems?"
    results = retrieve(query)

    for r in results:
        print("\n---")
        print("Score:", r["score"])
        print("Source:", r["source"])
        print("Text:", r["chunk"][:200])