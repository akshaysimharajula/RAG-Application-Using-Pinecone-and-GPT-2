# Retrieval Augmented Generation (RAG) Application

## Project Overview

This project implements a **Retrieval Augmented Generation (RAG)** system that allows users to upload a Q&A text document and ask questions based on its content. Instead of relying only on a language model’s internal knowledge, the system retrieves relevant information from the uploaded document and uses it as context to generate accurate answers.

The application converts document text into embeddings and stores them in a vector database. When a user asks a question, the system performs semantic search to retrieve the most relevant document chunks and then generates a response using a language model.

---

# Architecture

The project follows a **RAG pipeline**:

1. User uploads a document
2. Document is split into smaller chunks
3. Text chunks are converted into embeddings
4. Embeddings are stored in a vector database
5. User asks a question
6. Similar chunks are retrieved using vector search
7. Retrieved context is passed to the language model
8. The model generates the final answer

---

# Technologies Used

| Technology           | Purpose                       |
| -------------------- | ----------------------------- |
| Python               | Core programming language     |
| Streamlit            | Web application interface     |
| SentenceTransformers | Text embedding generation     |
| GPT-2                | Answer generation             |
| Pinecone             | Vector database for retrieval |
| NumPy                | Numerical processing          |

---

# Project Features

* Upload custom Q&A text documents
* Automatic text embedding generation
* Vector storage using Pinecone
* Semantic search for retrieving relevant context
* Answer generation using a transformer language model
* Interactive web interface

---

# Project Structure

```
rag-application
│
├── app.py
├── requirements.txt
├── data/
└── README.md
```

---

# Installation

## Clone Repository

```
git clone https://github.com/your-username/rag-application.git
cd rag-application
```

## Install Dependencies

```
pip install -r requirements.txt
```

## Run Application

```
streamlit run app.py
```

---

# Example Workflow

1. Upload a Q&A text file
2. Document embeddings are generated
3. Embeddings are stored in the vector database
4. User asks a question
5. Relevant context is retrieved
6. The language model generates the final answer

---

# Future Improvements

* Support for PDF and DOCX documents
* Multi-document retrieval
* Chat-based interface
* Integration with larger LLM models
* Improved prompt engineering

---

# Author

**Akshay**
Machine Learning / Generative AI Enthusiast
