
import requests
import numpy as np

from pathlib import Path


# CONFIGURATION

OLLAMA_URL = "http://localhost:11434"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"

KNOWLEDGE_DIR = Path("knowledge_base")

CHUNK_SIZE = 120
CHUNK_OVERLAP = 30

TOP_K = 3


# 1. LOAD DOCUMENTS

def load_documents():

    documents = []

    for file in KNOWLEDGE_DIR.glob("*.txt"):

        text = file.read_text(encoding="utf-8")

        documents.append({
            "source": file.name,
            "text": text
        })

    return documents


# 2. CHUNK DOCUMENTS

def chunk_documents(documents):

    chunks = []

    for document in documents:

        words = document["text"].split()

        start = 0

        while start < len(words):

            end = start + CHUNK_SIZE

            chunk_text = " ".join(words[start:end])

            chunks.append({
                "source": document["source"],
                "text": chunk_text
            })

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# 3. GENERATE EMBEDDINGS

def get_embeddings(texts):

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["embeddings"]


# 4. BUILD VECTOR STORE

def build_vector_store(chunks):

    texts = [chunk["text"] for chunk in chunks]

    embeddings = get_embeddings(texts)

    vector_store = []

    for chunk, embedding in zip(chunks, embeddings):

        vector_store.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "embedding": np.array(embedding)
        })

    return vector_store


# 5. COSINE SIMILARITY

def cosine_similarity(a, b):

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# 6. RETRIEVE RELEVANT CHUNKS

def retrieve(question, vector_store):

    query_embedding = get_embeddings([question])[0]

    query_vector = np.array(query_embedding)

    results = []

    for item in vector_store:

        score = cosine_similarity(
            query_vector,
            item["embedding"]
        )

        results.append({
            "text": item["text"],
            "source": item["source"],
            "score": score
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:TOP_K]


# 7. GENERATE ANSWER

def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(
        f"[{item['source']}] {item['text']}"
        for item in retrieved_chunks
    )

    prompt = f"""
You are a helpful college assistant.

Answer the question using ONLY the context below.

If the answer is not available in the context,
say "No idea."

Context:
{context}

Question: {question}

Answer:
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"]


# 8. MAIN FUNCTION

def main():

    print("\n===== MINI RAG SYSTEM =====\n")

    # Load documents
    documents = load_documents()

    if not documents:
        print("No documents found in knowledge_base/")
        return

    print(f"Loaded {len(documents)} documents.")

    # Chunk documents
    chunks = chunk_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    if not chunks:
        print("No text available in documents.")
        return

    # Build vector store
    print("\nGenerating embeddings...")

    vector_store = build_vector_store(chunks)

    print("Vector store created successfully!")

    print("\nRAG system is ready.")
    print("Type 'exit' to quit.\n")

    # Question-answer loop
    while True:

        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Exiting RAG system.")
            break

        if not question:
            continue

        # Retrieve relevant chunks
        retrieved_chunks = retrieve(
            question,
            vector_store
        )

        # Generate answer
        answer = generate_answer(
            question,
            retrieved_chunks
        )

        print("\nAssistant:", answer)

        print("\nSources:")

        for item in retrieved_chunks:
            print(
                f"- {item['source']} "
                f"(Similarity: {item['score']:.4f})"
            )

        print("\n" + "-" * 50 + "\n")


# ENTRY POINT

if __name__ == "__main__":
    main()
