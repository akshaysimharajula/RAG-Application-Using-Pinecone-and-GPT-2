# Retrieval Augmented Generation (RAG) Application Using Pinecone and GPT-2

## Overview

This project implements a **Retrieval Augmented Generation (RAG) Application Using Pinecone and GPT-2** that allows users to upload a Q&A text document and ask questions based on its content.

Unlike traditional language model applications that rely only on the model’s internal knowledge, this system first retrieves the most relevant information from the uploaded document and then generates an answer using that retrieved context. This improves the accuracy and relevance of the generated responses.

The application converts document text into embeddings using SentenceTransformers and stores them in a Pinecone vector database. When a user asks a question, the system performs semantic similarity search to retrieve the most relevant document chunks. These retrieved results are then passed as context to the GPT-2 language model to generate the final answer.

The entire application is built with Streamlit, providing a simple and interactive web interface where users can upload documents and ask questions easily.

## Features

* Upload Q&A text documents
* Generate embeddings from document content
* Store embeddings in Pinecone vector database
* Retrieve relevant information using semantic search
* Generate answers using GPT-2 model
* Simple and interactive Streamlit web interface

## Tech Stack

* Python
* Streamlit
* SentenceTransformers
* GPT-2
* Pinecone
* NumPy

## Project Structure

```
rag-application
│
├── app.py
├── requirements.txt
└── README.md
```

## Installation

### Clone Repository

```
git clone https://github.com/your-username/rag-application.git
cd rag-application
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Run Application

```
streamlit run app.py
```

## Usage

1. Upload a Q&A `.txt` document
2. The system converts the document into embeddings
3. Embeddings are stored in Pinecone
4. Enter a question related to the document
5. The system retrieves relevant context
6. GPT-2 generates the final answer

## Author

Akshay
