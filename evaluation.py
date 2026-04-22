import pandas as pd
from generator import generate_answer
from router import classify_query
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# MODEL (for evaluation metric)
# -----------------------------
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


# -----------------------------
# TEST DATA (15 queries)
# -----------------------------
test_data = [
    # FACTUAL
    {"query": "What are high-risk AI systems?", "type": "factual",
     "expected": "AI used in critical infrastructure, education, employment and law enforcement."},

    {"query": "What are penalties under EU AI Act?", "type": "factual",
     "expected": "Fines up to 35 million euros or 7 percent of global revenue."},

    {"query": "What does the US executive order require?", "type": "factual",
     "expected": "Safety testing and watermarking of AI generated content."},

    {"query": "What is frontier model threshold?", "type": "factual",
     "expected": "Models trained using more than 10^26 FLOPs."},

    {"query": "What is the role of EU AI Office?", "type": "factual",
     "expected": "Overseeing implementation and drafting standards."},

    # SYNTHESIS
    {"query": "Compare US and EU AI regulation approaches", "type": "synthesis",
     "expected": "EU uses comprehensive regulation while US uses sector specific flexible approach."},

    {"query": "What are common themes across AI regulations?", "type": "synthesis",
     "expected": "Transparency, accountability, data governance and human oversight."},

    {"query": "How do different countries regulate AI?", "type": "synthesis",
     "expected": "EU uses strict laws, US uses guidelines, China uses targeted regulations."},

    {"query": "What are challenges in global AI regulation?", "type": "synthesis",
     "expected": "Fragmentation and inconsistent definitions across regions."},

    {"query": "Compare penalties across regions", "type": "synthesis",
     "expected": "EU has strict fines, US has no unified penalties, China uses administrative penalties."},

    # OUT OF SCOPE
    {"query": "What is reinforcement learning?", "type": "out_of_scope", "expected": ""},
    {"query": "Who invented AI?", "type": "out_of_scope", "expected": ""},
    {"query": "Explain neural networks", "type": "out_of_scope", "expected": ""},
    {"query": "What is blockchain?", "type": "out_of_scope", "expected": ""},
    {"query": "What is Python programming?", "type": "out_of_scope", "expected": ""},
]


# -----------------------------
# COSINE SIMILARITY
# -----------------------------
def compute_similarity(ans, expected):
    if expected == "":
        return 0.0

    emb1 = model.encode([ans], normalize_embeddings=True)
    emb2 = model.encode([expected], normalize_embeddings=True)

    return float(cosine_similarity(emb1, emb2)[0][0])


# -----------------------------
# EVALUATION
# -----------------------------
def evaluate():
    results = []
    correct_routing = 0

    for item in test_data:
        query = item["query"]
        expected_type = item["type"]

        predicted_type, retrieved = classify_query(query)

        if predicted_type == expected_type:
            correct_routing += 1

        response = generate_answer(query)

        similarity = compute_similarity(response["answer"], item["expected"])

        results.append({
            "query": query,
            "expected_type": expected_type,
            "predicted_type": predicted_type,
            "routing_correct": predicted_type == expected_type,
            "answer_similarity": round(similarity, 3)
        })

    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    print("\n📊 RESULTS:\n")
    print(df)

    print("\n🎯 METRICS:")
    print("Routing Accuracy:", round(correct_routing / len(test_data), 2))

    return df


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    evaluate()