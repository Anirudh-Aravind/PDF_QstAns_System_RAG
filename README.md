# PDF Question-Answering System with RAG

A Retrieval Augmented Generation (RAG) system that allows users to upload PDFs and ask questions about their content. The system uses local LLMs via Ollama for both embeddings and question answering.

![image](https://github.com/user-attachments/assets/61812c93-cac3-4e26-a776-51fb6a41aa6c)
![image](https://github.com/user-attachments/assets/c8b7d106-cd25-4c22-9ec6-4a167e1d84e5)


## Features

- PDF document processing and chunking
- Vector storage using ChromaDB
- Local LLM integration via Ollama
- Streamlit web interface
- Object-oriented architecture
- Cross-encoder reranking

## Prerequisites

- Python 3.10+
- Ollama with llama2 model (llama3.2:3b)
- ChromaDB

- Ollama with required models:
```bash
# Install Ollama (one-time setup)
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull llama3.2:3b            # For text generation
ollama pull nomic-embed-text       # For embeddings
```
- ChromaDB
- GPU recommended but not required
  

## Installation

```bash
# Clone repository
git clone https://github.com/Anirudh-Aravind/rag_application.git
cd rag-application

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start Ollama service
ollama run llama3.2:3b
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Upload a PDF using the sidebar

4. Ask questions about the document content

## Project Structure

```
pdf-rag-system/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── rag-chroma/      # ChromaDB storage -- automatically create while the code execution
└── src/
    ├── document.py         # Document processing
    ├── vector_store.py     # Vector storage operations
    ├── llm.py              # LLM interface
    └── rag_app.py          # RAG application logic
```

## Configuration

Key parameters can be adjusted in their respective classes:

- `DocumentProcessor`: Chunk size, overlap
- `VectorStore`: Embedding model, collection name
- `LLMInterface`: Model name, system prompt
