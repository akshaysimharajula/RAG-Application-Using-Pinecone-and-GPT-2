# RAG Application using Pinecone

## Overview

This project implements a **Retrieval Augmented Generation (RAG)** system where users can upload a Q&A text document and ask questions related to its content. The system retrieves relevant information from the document and generates answers using a language model.

## Tech Stack

* Python
* Streamlit
* SentenceTransformers
* GPT-2
* Pinecone
* NumPy

## Features

* Upload Q&A text documents
* Generate embeddings from document text
* Store embeddings in Pinecone vector database
* Retrieve relevant context using semantic search
* Generate answers using GPT-2
* Simple web interface using Streamlit

## How to Run

```bash
git clone https://github.com/your-username/rag-application.git
cd rag-application
pip install -r requirements.txt
streamlit run app.py
```

## Author

Akshay
