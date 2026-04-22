from retriever import retrieve


# -----------------------------
# ROUTER LOGIC
# -----------------------------
def classify_query(query):
    results = retrieve(query, top_k=5)

    scores = [r["score"] for r in results]
    sources = [r["source"] for r in results]

    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    unique_sources = len(set(sources))

    # -----------------------------
    # 1. OUT OF SCOPE (more strict)
    # -----------------------------
    if max_score < 0.65:
        return "out_of_scope", results

    # -----------------------------
    # 2. FACTUAL
    # -----------------------------
    # one chunk clearly dominates
    if max_score > 0.85 and (max_score - avg_score) > 0.12:
        return "factual", results

    # -----------------------------
    # 3. SYNTHESIS
    # -----------------------------
    # multiple strong chunks
    if avg_score > 0.68:
        return "synthesis", results

    # fallback
    return "factual", results

# -----------------------------
# RESPONSE HANDLER
# -----------------------------
def generate_response(query):
    label, results = classify_query(query)

    if label == "out_of_scope":
        return {
            "type": label,
            "answer": "The provided documents do not contain enough information to answer this question."
        }

    elif label == "factual":
        return {
            "type": label,
            "answer": results[0]["chunk"]  # best chunk
        }

    elif label == "synthesis":
        combined = "\n\n".join([r["chunk"] for r in results[:3]])
        return {
            "type": label,
            "answer": combined
        }


# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    query = "Compare US and EU AI regulation approaches"

    response = generate_response(query)

    print("\nTYPE:", response["type"])
    print("\nANSWER:\n", response["answer"][:500])