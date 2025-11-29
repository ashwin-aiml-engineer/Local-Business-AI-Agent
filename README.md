# 🔒 Local Business AI Agent (Privacy-First RAG)

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red) ![Ollama](https://img.shields.io/badge/LLM-Llama3.2-orange) ![Status](https://img.shields.io/badge/Status-Active_Development-green)

**A production-ready, air-gapped AI Agent designed for secure business automation. It runs entirely on local hardware, ensuring no sensitive data ever leaves the premises.**

---

## 🚀 Project Overview
Businesses often deal with sensitive data (Legal Contracts, Financial Reports) that cannot be uploaded to public cloud models like ChatGPT.

This project solves that problem by deploying a **Local Retrieval Augmented Generation (RAG)** pipeline. It allows users to "Chat with their Documents" using an AI that runs 100% offline.

**Key Differentiator:** Optimized for standard consumer hardware (CPU Inference) to ensure stability on non-server grade laptops.

## ✨ Key Features
* **🛡️ 100% Privacy:** Zero data egress. Works without an internet connection.
* **🧠 RAG Architecture:** Ingests PDF documents and converts them into a Vector Database using **ChromaDB** and **Nomic Embeddings**.
* **🚫 Hallucination Control:** The "Lawyer Persona" is engineered to answer *only* based on the provided documents, citing specific sources (e.g., "Section 25F").
* **⚙️ Hardware Optimized:** Includes a custom "Batch Ingestion" pipeline to prevent thermal throttling on laptops with limited VRAM.

## 🛠️ Tech Stack
* **LLM Runtime:** [Ollama](https://ollama.com/) (Llama 3.2 - 3B Quantized)
* **Embeddings:** Nomic-Embed-Text
* **Orchestration:** LangChain Community
* **Vector DB:** ChromaDB (Persistent Storage)
* **Frontend:** Streamlit
* **Language:** Python 3.10

---

## 📖 How to Run Locally

### 1. Prerequisites
* Install [Ollama](https://ollama.com/).
* Pull the required models:
    ```bash
    ollama pull llama3.2
    ollama pull nomic-embed-text
    ```

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone [https://github.com/ashwin-aiml-engineer/Local-Business-AI-Agent.git](https://github.com/ashwin-aiml-engineer/Local-Business-AI-Agent.git)
cd Local-Business-AI-Agent

# Recommended: Create a virtual environment (Python 3.10)
pip install langchain langchain-community streamlit chromadb ollama pypdf
3. Ingest Your Data (Build the Brain)
Place your PDF document in the root folder and rename it to data.pdf.

Run the ingestion script to build the vector database:

Bash

python ingest.py
(Note: This uses a batched processing method to ensure CPU stability.)

4. Launch the Agent
Start the Web Interface:

Bash

streamlit run app.py
🧠 System Architecture
Ingestion: PDF -> Chunking (1000 chars) -> Nomic Embeddings -> ChromaDB.

Retrieval: User Query -> Similarity Search (Top 3 Chunks) -> Context Injection.

Generation: System Prompt (Strict Lawyer) + Context + Question -> Llama 3.2 -> Answer.

📅 Roadmap
[x] Day 33: System Prompts & Persona Engineering (Chef/Lawyer).

[x] Day 34: RAG Pipeline & PDF Ingestion.

[ ] Day 35: Chat History Memory (Multi-turn conversations).

[ ] Day 36: "Writer Agent" - Auto-drafting and saving legal notices to disk.

Author: Ashwin Part of the "Zero to AI Engineer" 90-Day Roadmap.