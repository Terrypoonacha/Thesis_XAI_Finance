import os
import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import google.generativeai as genai

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "knowledge"
VECTOR_DB_PATH = PROJECT_ROOT / "data" / "vector_db"

def ingest_documents():
    # 1. Check for API Key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not found.")
        print("Please set your Google API key to proceed.")
        sys.exit(1)
    
    # Configure GenAI
    genai.configure(api_key=api_key)

    # 2. Locate PDFs
    pdf_files = list(KNOWLEDGE_PATH.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {KNOWLEDGE_PATH}")
        sys.exit(1)
        
    print(f"Found {len(pdf_files)} PDFs: {[f.name for f in pdf_files]}")

    all_docs = []
    
    # 3. Load PDFs
    for pdf_path in pdf_files:
        print(f"Loading {pdf_path.name}...")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        all_docs.extend(docs)
        print(f"Loaded {len(docs)} pages from {pdf_path.name}")

    # 4. Split Text
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(all_docs)
    print(f"Created {len(splits)} chunks.")

    # 5. Generate Embeddings & Store in ChromaDB
    print("Generating embeddings and storing in ChromaDB...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Create valid persistent directory
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_PATH)
    )

    import time
    batch_size = 1
    # Test with just 5 chunks to verify access
    splits = splits[:5] 
    total_chunks = len(splits)
    print(f"TEST MODE: Ingesting {total_chunks} chunks in batches of {batch_size}...", flush=True)

    for i in range(0, total_chunks, batch_size):
        batch = splits[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}...", flush=True)
        try:
            vector_store.add_documents(batch)
            time.sleep(10) # Aggressive rate limit buffer
        except Exception as e:
            print(f"Error adding batch {i}: {e}")
            time.sleep(10) # Backoff
            try:
                vector_store.add_documents(batch)
            except Exception as e2:
                 print(f"Failed retry for batch {i}: {e2}")

    print(f"Vector Database initialized at {VECTOR_DB_PATH}")

    # 6. Validation
    print("\n--- Validation: Similarity Search ---")
    query = "Article 13 Transparency"
    print(f"Query: '{query}'")
    results = vector_store.similarity_search(query, k=3)
    
    for i, res in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Source: {res.metadata.get('source', 'Unknown')} (Page {res.metadata.get('page', 'Unknown')})")
        print(f"Content Snippet: {res.page_content[:200]}...")

if __name__ == "__main__":
    ingest_documents()
