# Trivia-RAG-Explorer 🧠🔍
An advanced Retrieval-Augmented Generation (RAG) system built to answer complex open-domain questions from the **TriviaQA** dataset using a local LLM pipeline.

## 🚀 Overview
This project implements a complete RAG pipeline that leverages:
- **LLM:** Mistral-7B (via HuggingFace)
- **Vector Database:** ChromaDB
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Framework:** LangChain

The system is optimized to process long-form documents (some up to 89k words) and retrieve the most relevant context to provide accurate, evidence-based answers.

---

## 📊 Data Analysis & Chunking Strategy
Before building the vector store, a deep statistical analysis was performed on the raw paragraphs to optimize retrieval.

### 1. Original Data Insights
- **Total Paragraphs:** 5,541
- **Median Length:** 1,038 words.
- **Complexity:** 90% of documents are under 6,098 words, but the **longest document reached 89,615 words**, requiring a robust chunking strategy.

### 2. Optimized Chunking
To stay within the model's token limits while preserving context:
- **Total Chunks:** 128,151
- **Average Chunk Size:** ~158 words.
- **P90 Chunk Size:** 193 words (Optimized for `all-MiniLM-L6-v2` 256-token limit).
- **Max Chunk Size:** 234 words.

---

## 🛠️ Data Cleaning & Pre-processing
To ensure high-quality retrieval and reduce noise, a specialized `clean_text` pipeline was implemented.

### The Cleaning Logic:
- **Normalization:** Removed extra whitespaces and unified line breaks to improve embedding consistency.
- **Semantic Thresholding:** A **15-character minimum limit** was enforced.
- **Why?** This effectively prunes "dead data" (e.g., page numbers, fragmented headers, or metadata like "[Edit]") while strictly preserving short, factual sentences critical for Trivia questions.

---

## 🏗️ System Architecture
The system follows a classic RAG architecture:
1. **Retrieval:** Uses Similarity Search to fetch the top 5 (k=5) most relevant chunks from ChromaDB.
2. **Augmentation:** Injecting the retrieved context into a custom-engineered prompt template.
3. **Generation:** Mistral-7B generates a concise answer based *only* on the provided context.

### Custom Prompt Design:
```text
<s>[INST] You are a helpful QA assistant. Use the following context to answer the question accurately. 
Keep the answer concise and direct. If the answer is not in the context, say "Not found in context".
⚙️ Configuration
The LLM pipeline is tuned for high precision:

Temperature: 0.1 (To ensure deterministic and factual responses).

Max New Tokens: 256.

Sampling: Disabled (do_sample=False) to minimize hallucinations.

📂 Project Structure
pipeline.py: The core RAG execution script.

clean_text.py: Text pre-processing and filtering utility.

chroma_db/: Persisted vector database (Generated after indexing).

requirements.txt: List of necessary Python libraries.

🚦 How to Run
Clone the repository.

Install dependencies:

Bash

pip install -r requirements.txt
Run the pipeline:

Bash

python src/pipeline.py
Context: {context}
Question: {question} [/INST]
Answer:

# TriviaQA RAG Project

This repository contains a Retrieval-Augmented Generation (RAG) system for TriviaQA questions. The pipeline leverages **sentence embeddings**, a **vector store (ChromaDB)**, and a **Mistral-7B LLM** for short, context-based answers.

---

## 📦 Repository Structure

```
├── requirements.txt         # Python dependencies
├── embedding_model.py       # Sentence embedding model setup
├── llm_model.py             # LLM (Mistral-7B) setup
├── preprocess.py            # TriviaQA preprocessing & chunking
├── vector_store.py          # Build vector DB from preprocessed chunks
├── pipeline.py              # RAG pipeline & run_rag function
├── app.py                   # FastAPI wrapper for API queries
├── evaluate.py              # Evaluation script for accuracy & latency
└── README.md                # This file
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**

```bash
git clone <repo_url>
cd <repo_folder>
```

2. **Set your HuggingFace token**

```bash
export HF_TOKEN=<your_hf_token>
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Preprocess TriviaQA dataset**

```bash
python preprocess.py
```

This generates `chunks_metadata.json`.

5. **Build the vector store**

```bash
python vector_store.py
```

Embeddings are stored in `/kaggle/working/chroma_db`.

6. **Run the RAG pipeline interactively**

```bash
python pipeline.py
```

Or start the FastAPI server:

```bash
python app.py
```

Query via POST request to `http://127.0.0.1:8050/query` with JSON payload `{"question": "Your question here"}`.

7. **Run evaluation**

```bash
python evaluate.py
```

Generates `evaluation_results.csv` with accuracy and latency.

---

## 🧩 Pipeline Architecture

```
[TriviaQA Raw Dataset]
        |
        v
[preprocess.py] -- Clean + Chunk --> [chunks_metadata.json]
        |
        v
[vector_store.py] -- Embed --> [Chroma Vector DB]
        |
        v
[pipeline.py] -- run_rag() --> [RAG Output]
        |
        v
[app.py / API] --> [Answer + Context + Latency]
```

**Notes:**

* LLM answers are STRICTLY context-based (1–3 words).
* If the answer is not found, the system returns: `Not found in context`.
* Empty or whitespace queries return: `Empty query received`.

---

## 📊 Accuracy & Performance Summary

* **Chunking Stats:**

  * Original dataset P90: ~6,098 words per paragraph
  * After chunking: ~193 words per chunk (≈31x reduction)
  * Total chunks generated: ~128k

* **Evaluation on 20 TriviaQA questions (example subset):**

  * Accuracy (Correct + Partially Correct): computed dynamically in `evaluate.py`
  * Average response latency: measured per query by `run_rag`

* **Key design decisions:**

  * Whitespace normalization ensures semantic embedding quality.
  * Minimum 15-character threshold removes trivial metadata.
  * 50–256 token chunks with 50-token overlap maintain context fidelity.

---

## 🚀 Usage Notes

* Use GPU if available for faster embedding and LLM inference.
* FastAPI server allows external integration.
* All scripts are modular and can be run independently.
* Dockerfile can be added for containerized deployment.

---

**Author:** Your Name
**Date:** 2025-12-26
