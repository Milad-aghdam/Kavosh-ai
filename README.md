Here is the complete, professional `README.md` file. It includes the required metadata for Hugging Face, the architecture section, and clear instructions.

Copy and paste this entirely into your `README.md` file.

```markdown
---
title: Kavosh AI
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# 📚 Kavosh-AI: Multilingual Academic Q&A System

Kavosh-AI is a smart, multilingual research assistant that extracts precise, cited answers from PDF documents. Ask questions in English or Persian (Farsi) and get answers based directly on the document's content, complete with source references.

![Kavosh-AI Demo](./assets/kavosh-ai-demo.png)

---

## 🏗️ System Architecture

This project implements an **Advanced RAG (Retrieval-Augmented Generation)** pipeline designed for high precision and multilingual support.

**Key Architectural Components:**
1.  **Hybrid Search:** Combines **Dense Retrieval** (FAISS + BGE-M3 Embeddings) for semantic understanding and **Sparse Retrieval** (BM25) for keyword matching.
2.  **Reranking:** Utilizes a Cross-Encoder (`BAAI/bge-reranker-v2-m3`) to re-score and filter the top retrieved documents, significantly reducing hallucinations.
3.  **Multilingual Core:** Built natively to handle English and Persian queries interchangeably.

![Architecture Diagram](./assets/architecture_diagram.png)

---

## ✨ Features

- **Advanced RAG Pipeline:** Uses Hybrid Search (Vector + Keyword) to find the most relevant context.
- **Cross-Encoder Reranking:** A "Judge" model evaluates retrieved data to ensure only high-quality context reaches the LLM.
- **Source Citations:** Every answer is backed by direct quotes from the source document for verification.
- **Multilingual Support:** Seamlessly ask questions in English or Persian.
- **Session Management:** Supports multiple concurrent users via isolated session states.
- **Containerized:** Fully containerized with Docker for easy deployment and reproducibility.

## 🛠️ Tech Stack

- **Backend:** Python 3.10
- **Orchestration:** LangChain
- **Embeddings:** `BAAI/bge-m3` (State-of-the-art Multilingual Model)
- **Retrieval:** FAISS (Dense) + Rank-BM25 (Sparse)
- **Reranking:** `BAAI/bge-reranker-v2-m3`
- **LLM Integration:** OpenRouter API (Mistral-7B / Llama-3)
- **Web UI:** Gradio
- **DevOps:** Docker, GitHub Actions (CI/CD)

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Git
- Docker (Optional, for containerized run)
- An API key from [OpenRouter.ai](https://openrouter.ai/)

### 2. Local Installation

Clone the repository:
```bash
git clone https://github.com/YOUR-USERNAME/KAVOSH-AI.git
cd KAVOSH-AI
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create a `.env` file in the root directory and add your API key:
```bash
OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

Run the application:
```bash
python app.py
```

### 3. Running with Docker

Make sure you have Docker Desktop installed and running.

Build the Docker image:
```bash
docker build -t kavosh-ai .
```

Run the container:
```bash
docker run -p 7860:7860 -e OPENROUTER_API_KEY="your_openrouter_api_key_here" kavosh-ai
```

Navigate to `http://localhost:7860` in your browser.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
```