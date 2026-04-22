# Failure Analysis

## Overview

This document outlines key failure cases observed in the Agentic RAG system. While the system performs well on direct factual queries, it shows limitations in routing accuracy, multi-document synthesis, and handling contradictory information.

---

## Failure Case 1: Incorrect Routing (Factual vs Synthesis)

**Query:**  
Compare US and EU AI regulation approaches

**Expected:**  
Synthesis

**Observed:**  
Factual

**Issue:**  
The router relies heavily on similarity scores. If one chunk has a high score, the query is incorrectly classified as factual.

**Root Cause:**  
Lack of source diversity awareness in routing logic.

**Improvement:**  
Incorporate source-based heuristics. If retrieved chunks come from multiple documents, classify as synthesis.

---

## Failure Case 2: Weak Synthesis

**Query:**  
How do different countries regulate AI?

**Expected:**  
Combined answer using multiple sources

**Observed:**  
Fragmented or loosely connected response

**Issue:**  
Chunks are concatenated without structured reasoning.

**Root Cause:**  
The generator prompt does not explicitly guide synthesis.

**Improvement:**  
Enhance prompt to include instructions like "compare", "summarize differences", and "combine insights".

---

## Failure Case 3: Handling Contradictory Information

**Example:**  
Different documents report different penalty values under the EU AI Act.

**Expected:**  
Mention both values and highlight inconsistency

**Observed:**  
Only one value is returned

**Issue:**  
The system does not detect contradictions.

**Root Cause:**  
No contradiction-aware logic in retrieval or generation.

**Improvement:**  
Modify prompt to explicitly mention conflicting information when present.

---

## Failure Case 4: Out-of-Scope Misclassification

**Query:**  
What is machine learning?

**Expected:**  
Out of scope

**Observed:**  
Classified as factual

**Issue:**  
Semantic similarity retrieves loosely related chunks.

**Root Cause:**  
Embedding similarity does not guarantee relevance.

**Improvement:**  
Increase threshold and add domain filtering or intent detection.

---

## Failure Case 5: Chunk Boundary Issues

**Issue:**  
Important information is split across chunks.

**Observed:**  
Incomplete or partial answers

**Root Cause:**  
Fixed-size chunking without full semantic awareness.

**Improvement:**  
Adopt sentence-based or recursive chunking strategies.

---

## Summary of Limitations

- Routing depends on simple heuristics  
- No contradiction handling  
- Limited synthesis capability  
- Chunking may break context  
- Evaluation limited to cosine similarity  

---

## Future Improvements

- Add reranking (cross-encoder)  
- Improve routing with query intent detection  
- Enhance prompts for synthesis and contradiction awareness  
- Use hybrid retrieval (semantic + keyword)  
- Explore additional evaluation metrics  

---

## Conclusion

The system demonstrates strong performance for factual retrieval but highlights challenges in multi-document reasoning and robustness. These limitations provide clear directions for future improvements in building more reliable agentic RAG systems.