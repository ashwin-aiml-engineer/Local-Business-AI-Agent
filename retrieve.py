from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import sys
import os

# --- CONFIGURATION ---
# We use the exact same model you used to create the database
# based on your app.py code, you used 'nomic-embed-text'
EMBEDDING_MODEL = "nomic-embed-text"
DB_FOLDER = "vector_db"

def main():
    # 1. VERIFY FOLDER
    if not os.path.exists(DB_FOLDER):
        print(f"❌ CRITICAL ERROR: The folder '{DB_FOLDER}' does not exist.")
        print("   Please run your ingestion script first!")
        return

    # 2. LOAD THE BRAIN
    print(f"🔌 Connecting to Vector Database in '{DB_FOLDER}'...")
    print(f"   Using Embedding Model: {EMBEDDING_MODEL}")
    
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_store = Chroma(
            persist_directory=DB_FOLDER,
            embedding_function=embeddings
        )
        print("✅ Database Loaded Successfully!")
    except Exception as e:
        print(f"❌ Failed to load database: {e}")
        return

    # 3. TEST SEARCH (The "Needle in Haystack" Test)
    # This specific question tests if it knows about Section 25F from your PDF
    test_query = "What are the conditions for retrenchment under Section 25F?"
    
    print(f"\n🔎 RUNNING TEST QUERY: '{test_query}'")
    results = vector_store.similarity_search(test_query, k=3)

    if not results:
        print("❌ RESULT: No matching text found. The database might be empty.")
    else:
        print(f"✅ RESULT: Found {len(results)} matches from the PDF.\n")
        for i, doc in enumerate(results):
            print(f"--- MATCH {i+1} ---")
            # We print the raw text found in the PDF
            print(f"📄 TEXT: {doc.page_content[:300]}...") 
            print(f"📍 SOURCE: Page {doc.metadata.get('page', 'Unknown')}")
            print("-" * 30)

    # 4. INTERACTIVE MODE
    print("\n💡 SYSTEM READY. Type a legal question to search the Act.")
    while True:
        user_q = input("\n🗣️ Enter Query (or 'exit'): ")
        if user_q.lower() == 'exit':
            break
        
        results = vector_store.similarity_search(user_q, k=3)
        if results:
            print(f"   Found match on Page {results[0].metadata.get('page', 'Unknown')}:")
            print(f"   \"{results[0].page_content[:200]}...\"")
        else:
            print("   No relevant text found.")

if __name__ == "__main__":
    main()