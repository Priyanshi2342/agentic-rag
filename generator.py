import os
import requests
from dotenv import load_dotenv
from router import classify_query

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY not found in .env file")


# -----------------------------
# LLM CALL FUNCTION
# -----------------------------
def call_llm(context, query):
    prompt = f"""
You are a strict QA system.

Rules:
- Answer ONLY using the provided context
- If answer is not clearly present, say: "Information not available."
- Do NOT use outside knowledge
- Be concise and factual

Context:
{context}

Question:
{query}
"""

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "meta-llama/llama-3-8b-instruct",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    result = response.json()

    if "choices" not in result:
        return "Error: LLM response failed."

    return result["choices"][0]["message"]["content"]


# -----------------------------
# GENERATOR (MAIN FUNCTION)
# -----------------------------
def generate_answer(query):
    label, results = classify_query(query)

    #  OUT OF SCOPE → no LLM call
    if label == "out_of_scope":
        return {
            "type": label,
            "answer": "Information not available."
        }

    # build context from top chunks
    context = "\n\n".join([r["chunk"] for r in results[:3]])

    answer = call_llm(context, query)

    return {
        "type": label,
        "answer": answer
    }


# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    while True:
        query = input("\nEnter your query (or type 'exit'): ")

        if query.lower() == "exit":
            break

        response = generate_answer(query)

        print("\n-----------------------------")
        print("TYPE:", response["type"])
        print("ANSWER:", response["answer"])