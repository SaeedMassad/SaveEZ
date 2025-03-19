import os
import nltk
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document  # ✅ Make sure this import matches your version

# Ensure necessary NLTK resources are available
nltk.download('punkt', download_dir='/Users/saeedmassad/Desktop/Honours Project/env/nltk_data')
 # Sentence tokenizer

def semantic_chunking(text, max_chunk_size=500):
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Load PDFs
pdf_directory = "/Users/saeedmassad/Desktop/Honours Project/data/Books"
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]

all_documents = []
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    print(f"Processing: {pdf_file}")

    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    all_documents.extend(documents)

# Semantic chunking
all_chunks = []
for doc in all_documents:
    text_chunks = semantic_chunking(doc.page_content, max_chunk_size=500)
    for chunk in text_chunks:
        # Create Document instances instead of dicts
        document = Document(page_content=chunk, metadata=doc.metadata)
        all_chunks.append(document)

# Initialize Chroma
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# Add Documents
chroma_db.add_documents(all_chunks)

print(f"{len(all_chunks)} semantic chunks added to ChromaDB.")

