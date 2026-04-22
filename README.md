# 🤖 Agentic RAG System with Evaluation

## 📌 Overview

This project implements an **Agentic Retrieval-Augmented Generation (RAG)** system over a collection of AI regulation documents.  
The system is designed to not only retrieve relevant information but also **decide how to answer a query** using explicit routing logic.

Unlike standard RAG pipelines, this system introduces an **agentic router** that classifies queries into:

- **Factual** → Answerable from a single source  
- **Synthesis** → Requires combining information from multiple sources  
- **Out-of-Scope** → Not answerable from the given documents  

The system is fully evaluated using a custom evaluation pipeline and includes detailed failure analysis.

---

# 🏗️ System Architecture

The pipeline consists of five core components:
User Query
↓
Retriever (FAISS + Embeddings)
↓
Agentic Router (Decision Logic)
↓
Generator (LLM)
↓
Final Answer


---

## 1. 📄 Ingestion Pipeline (`ingestion.py`)

### 🔹 Document Loading
- Documents are loaded from the `/data` directory
- Each document is tagged with metadata (source file)

---

### 🔹 Chunking Strategy

The system uses a **fixed-size overlapping chunking approach**:

- **Chunk Size:** ~200 words  
- **Overlap:** ~40 words  

#### ✅ Why this strategy?

- Prevents **context loss at boundaries**
- Maintains **semantic coherence**
- Improves retrieval accuracy by preserving partial overlaps
- Works well for:
  - policy documents  
  - technical briefs  
  - news-style content  

---

### 🔹 Embedding Model
BAAI/bge-small-en-v1.5


#### ✅ Why BGE?

- High performance for **semantic similarity tasks**
- Better than MiniLM for **retrieval tasks**
- Produces meaningful similarity scores even for complex queries
- Lightweight and fast (suitable for local execution)

---

### 🔹 Vector Store

- **FAISS (IndexFlatL2)**
- Enables fast nearest-neighbor search over embeddings

---

## 2. 🔎 Retriever (`retriever.py`)

- Converts query → embedding  
- Performs **Top-K (k=5) retrieval**
- Uses **cosine similarity (via L2 distance equivalence)**  

Returns:
- relevant chunks  
- similarity scores  
- source metadata  

---

## 3. 🧠 Agentic Router (`router.py`)

This is the **core intelligence layer** of the system.

> ⚠️ Important: No LLM is used for routing (as per assignment requirement)

---

### 🔹 Routing Strategy

The router uses **heuristic-based decision logic** based on:

- Maximum similarity score  
- Average similarity score  
- Score distribution  

---

### 🔹 Final Routing Logic

```python
if max_score < 0.65:
    return "out_of_scope"

elif max_score > 0.85 and (max_score - avg_score) > 0.12:
    return "factual"

elif avg_score > 0.68:
    return "synthesis"

else:
    return "factual"
## 🔹 Challenges

- Embedding similarity is not perfect  
- Even unrelated queries can have moderate scores  
- Overlapping content causes ambiguity  

---

## 4. 💬 Generator (`generator.py`)

The generator uses an LLM via OpenRouter:
meta-llama/llama-3-8b-instruct


### 🔹 Prompt Design

The system enforces strict rules:

- Answer ONLY using provided context  
- Do NOT use external knowledge  
- If answer not found → return:  
Information not available.


---

### 🔹 Purpose

- Ensures grounded answers  
- Prevents hallucination  
- Improves reliability  

---

## 5. 📊 Evaluation Framework (`evaluation.py`)

### 🔹 Test Dataset

- Total: 15 queries  
- 5 factual  
- 5 synthesis  
- 5 out-of-scope  

---

### 🔹 Metrics Used

#### 1. Routing Accuracy
- Measures classification correctness  

#### 2. Answer Similarity
- Cosine similarity between:
- generated answer  
- expected answer  

---

### 🔹 Output

- Printed in terminal  
- Saved to:
results.csv


---

## 📊 Results Summary

- Routing Accuracy: ~0.67  

---

### 🔹 Observations

#### ✅ Strong Performance:
- Synthesis queries handled well  
- Multi-document reasoning effective  

#### ⚠️ Limitations:
- Factual queries sometimes misclassified  
- Out-of-scope detection imperfect  
- Embedding similarity can be misleading  

---

## ⚠️ Failure Analysis

Detailed analysis is provided in:
FAILURE.md

---

### 🔹 Key Failure Cases

1. **Factual vs Synthesis Confusion**  
   - Due to overlapping information across documents  

2. **Out-of-Scope Misclassification**  
   - Embeddings produce non-zero similarity  

3. **Chunk Boundary Issues**  
   - Context split across chunks  

---

## 🛠️ Tech Stack

- Python  
- SentenceTransformers  
- FAISS  
- OpenRouter API  
- scikit-learn  
- NumPy  
- Pandas  

---

## ▶️ How to Run

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt

python ingestion.py
python evaluation.py

## 📂 Project Structure
agentic-rag/
│
├── data/
├── ingestion.py
├── retriever.py
├── router.py
├── generator.py
├── evaluation.py
├── results.csv
├── FAILURE.md
├── requirements.txt
└── README.md

##Video Demo Link:
https://drive.google.com/file/d/1Kc9v6XahYUvrOhKIe9dJGZ7_O8KnOjNw/view?usp=drive_link
