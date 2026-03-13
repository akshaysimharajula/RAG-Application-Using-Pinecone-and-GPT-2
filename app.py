import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pinecone import Pinecone, ServerlessSpec

# -------------------- CONFIG --------------------

PINECONE_API_KEY = "pcsk_73bFc1_A619NWNn74AbCQoQqa7XE9hW3iUQAdHt4tA5MbxgxLBhZhKWgyYBpi8s344RyWP"
INDEX_NAME = "rag-index"

# -------------------- LOAD MODELS --------------------

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline("text-generation", model="gpt2")
    return embed_model, generator

# -------------------- LOAD DOCUMENTS --------------------

@st.cache_data
def load_documents(uploaded_file):
    text = uploaded_file.read().decode("utf-8")
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    documents = [
        f"{lines[i]} {lines[i+1]}"
        for i in range(0, len(lines) - 1, 2)
    ]
    return documents

# -------------------- PINECONE SETUP --------------------

@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    index_names = [idx["name"] for idx in pc.list_indexes()]

    if INDEX_NAME not in index_names:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    return pc.Index(INDEX_NAME)

# -------------------- STORE EMBEDDINGS --------------------

def store_embeddings(index, documents, embed_model):
    embeddings = embed_model.encode(documents).tolist()

    vectors = [
        {
            "id": str(i),
            "values": embeddings[i],
            "metadata": {"text": documents[i]}
        }
        for i in range(len(documents))
    ]

    index.upsert(vectors=vectors)

# -------------------- RETRIEVE CONTEXT --------------------

def retrieve_context(query, embed_model, index, top_k=2):
    query_vector = embed_model.encode(query).tolist()

    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    return "\n".join(
        match["metadata"]["text"]
        for match in result["matches"]
    )

# -------------------- GENERATE ANSWER --------------------

def generate_answer(query, context, generator):
    prompt = f"""
Use ONLY the information given in the context below.
Do NOT repeat the question.
Do NOT ask new questions.
If the answer is not present, say "I don't know."

Context:
{context}

Question:
{query}

Answer:
"""

    output = generator(
        prompt,
        max_new_tokens=160,     # allows multi-line answer
        temperature=0.2,
        do_sample=False,
        eos_token_id=generator.tokenizer.eos_token_id
    )

    text = output[0]["generated_text"]

    # Extract text strictly after "Answer:"
    answer = text.split("Answer:")[-1].strip()

    # Stop if model starts repeating the question
    stop_phrases = [
        "Question:",
        "Context:",
        query
    ]

    for phrase in stop_phrases:
        if phrase in answer:
            answer = answer.split(phrase)[0].strip()

    return answer

# -------------------- STREAMLIT APP --------------------

def main():
    st.set_page_config(page_title="RAG App", layout="centered")
    st.title("📘 RAG App (Pinecone + GPT-2)")

    uploaded_file = st.file_uploader("Upload Q&A file", type="txt")

    if uploaded_file:
        embed_model, generator = load_models()
        index = init_pinecone()

        documents = load_documents(uploaded_file)
        store_embeddings(index, documents, embed_model)

        st.success("Documents indexed successfully")

        query = st.text_input("Ask a question")

        if query:
            context = retrieve_context(query, embed_model, index)
            answer = generate_answer(query, context, generator)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieved Context")
            st.code(context)

if __name__ == "__main__":
    main()