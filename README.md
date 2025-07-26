# resume-rag

# Resume RAG System

An end-to-end Retrieval-Augmented Generation system for querying resume content.

## System Components
1. **Retriever**: 
   - Uses FAISS vector store with SentenceTransformers embeddings
   - Finds relevant resume sections based on semantic similarity
2. **Generator**: 
   - DeepSeek-7B-Chat model from Hugging Face
   - Generates answers using retrieved context and conversation history

## Local Setup
1. Create virtual environment:
```bash
python -m venv rag-env
source rag-env/bin/activate