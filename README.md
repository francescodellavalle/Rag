# Rag for Context-Grounded QA Assistant

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![LangChain](https://img.shields.io/badge/orchestrator-LangChain-green.svg)
![ChromaDB](https://img.shields.io/badge/vectorstore-ChromaDB-purple.svg)
![LLM](https://img.shields.io/badge/model-Qwen2.5--3B--Instruct-orange.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

A clean step-by-step implementation of a fully local Retrieval-Augmented Generation (**RAG**) pipeline. Designed to run seamlessly inside Google Colab (T4 GPU), this project demonstrates how to ingest, index, query, and evaluate custom documents using open-source models without relying on third-party serverless API calls.

---

## 🛠️ Tech Stack & Architecture

- **Orchestration:** LangChain (LCEL)
- **Vector Database:** ChromaDB (In-memory)
- **Embedding Model:** Hugging Face `sentence-transformers/all-MiniLM-L6-v2`
- **Local LLM:** `Qwen/Qwen2.5-3B-Instruct` (quantized to 4-bit via `bitsandbytes`)
- **Evaluation:** Dynamic Reference-Free LLM-as-a-Judge

---

## 🚀 Key Features

1. **100% Local Pipeline:** Operates entirely locally on a free Colab T4 GPU, eliminating external API routing errors.
2. **Semantic Chunking:** Leverages `RecursiveCharacterTextSplitter` to preserve contextual boundaries.
3. **Optimized 4-Bit Inference:** Loads `Qwen2.5-3B-Instruct` in 4-bit precision (`nf4`).
4. **Strict Grounding:** Enforces strict anti-hallucination prompts paired with source citations for full transparency.
5. **Dynamic Evaluation (LLM-as-a-Judge):** Includes a reference-free evaluator auditing outputs on-the-fly for **Faithfulness** and **Answer Relevance**.

---

## 📋 Quick Start

### Running in Google Colab

1. Open the notebook in Google Colab (ensure runtime GPU is set to **T4**).
2. Execute the cells sequentially:
   - **Section 1:** Document Ingestion & Chunking
   - **Section 2:** Vector Indexing with ChromaDB
   - **Section 3:** Retriever Setup
   - **Section 4:** Local Model Loading & LCEL Chain Execution
   - **Section 5:** Dynamic LLM Evaluation

---

## 📊 Evaluation Results

The pipeline includes a self-auditing function evaluating output quality on custom user input:

```text
- Faithfulness Score: 5/5 - The response is fully grounded in the provided context.
- Answer Relevance Score: 5/5 - The response directly addresses the user query.
