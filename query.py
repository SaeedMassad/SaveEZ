from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from llama_cpp import Llama

model_path = "/Users/saeedmassad/Desktop/Honours Project/models/llama-2-7b.Q5_K_M.gguf"

# Initialize the embedding function
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def handle_query(query, chroma_db_path: str, model) -> str:
    """
    Handles a query by retrieving relevant documents from ChromaDB
    and generating an answer using the provided Llama model.
    """
    from langchain_chroma import Chroma
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings

    # Load ChromaDB
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chroma_db = Chroma(persist_directory=chroma_db_path, embedding_function=embedding)

    # Retrieve relevant documents
    results = chroma_db.similarity_search(query, k=5)
    relevant_documents = [result.page_content for result in results]

    # Combine into context
    context = " ".join(relevant_documents) if relevant_documents else "No relevant information found."

    # Build the prompt
    prompt = f"""
    You are SaveEZ, an AI assistant specializing in financial literacy.  
    Your goal is to provide clear, accurate, and educational responses **only using** the provided context.  

    Context:
    {context}

    Question: {query}

    Instructions:
    - Answer based only on the context above.
    - If the context lacks enough info, say: "I don't have enough information to answer that."

    Answer:
    """

    # Generate response
    response = model(
        prompt=prompt,
        max_tokens=300,
        temperature=0.3
    )

    # Extract and return the response text
    return response["choices"][0]["text"]


