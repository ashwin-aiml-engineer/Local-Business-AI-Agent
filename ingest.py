import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# 1. SETUP
print("--- STARTING BATCH INGESTION ---")
# This uses the embedding model we downloaded
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. LOAD PDF
pdf_file_path = "data.pdf"
if not os.path.exists(pdf_file_path):
    print("ERROR: data.pdf not found!")
    exit()

print("1. Loading PDF...")
loader = PyPDFLoader(pdf_file_path)
docs = loader.load()
print(f"   -> Loaded {len(docs)} pages.")

# 3. SPLIT TEXT
print("2. Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)
print(f"   -> Created {len(chunks)} chunks.")

# 4. BATCH SAVING (The Fix)
print("3. Saving to Database (Batch Mode)...")

# Process in small batches of 5 to prevent CPU hang
batch_size = 5
total_chunks = len(chunks)

# Initialize Database with the first batch
print(f"   Initializing DB with first {batch_size} chunks...")
vector_store = Chroma.from_documents(
    documents=chunks[:batch_size],
    embedding=embeddings,
    persist_directory="vector_db"
)

# Loop through the rest
for i in range(batch_size, total_chunks, batch_size):
    # Get a slice of 5 chunks
    batch = chunks[i : i + batch_size]
    
    # Add to existing DB
    vector_store.add_documents(batch)
    
    # Progress Report
    print(f"   Processed {min(i + batch_size, total_chunks)}/{total_chunks} chunks...")
    
    # Small sleep to let CPU breathe
    time.sleep(0.1)

print("✅ SUCCESS! Database created.")
