# 📚 Kavosh-AI: Multilingual Academic Q&A System

Kavosh-AI is a smart, multilingual research assistant that extracts precise, cited answers from PDF documents. Ask questions in English or Persian (Farsi) and get answers based directly on the document's content, complete with source references.

![Kavosh-AI Demo](./assets/kavosh-ai-demo.png)
---

## ✨ Features

- **Retrieval-Augmented Generation (RAG):** Provides answers based on document content, not pre-trained knowledge, reducing hallucinations.
- **Source Citations:** Every answer is backed by direct quotes from the source document for verification.
- **Multilingual Support:** Seamlessly ask questions in English or Persian.
- **Interactive Web UI:** Easy-to-use interface built with Gradio for uploading PDFs and asking questions.
- **Containerized:** Fully containerized with Docker for easy deployment and portability.

## 🛠️ Tech Stack

- **Backend:** Python
- **AI/ML Frameworks:** LangChain, Sentence-Transformers (for embeddings), FAISS (for vector search)
- **LLM Integration:** Connects to free, high-performance models via OpenRouter.
- **Web UI:** Gradio
- **Containerization:** Docker

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- Git
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


Create a .env file in the root directory and add your API key:
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
Navigate to http://localhost:7860 in your browser.