import streamlit as st
import ollama
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# --- 1. CONFIGURATION (CPU SAFE MODE) ---
# Prevent Blue Screen by forcing CPU only
os.environ["OLLAMA_NUM_GPU"] = "0"

st.set_page_config(page_title="RAG Lawyer", page_icon="⚖")
st.title("⚖ AI Corporate Lawyer (RAG)")
st.caption("Powered by Llama 3.2 + Industrial Disputes Act 1947")

# --- 2. LOAD THE KNOWLEDGE BASE ---
# We use @st.cache_resource so it only loads the DB once, not every time you type.
@st.cache_resource
def load_vector_db():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        persist_directory="vector_db",
        embedding_function=embeddings
    )
    return vector_store

# Load the database now
vector_db = load_vector_db()

# --- 3. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "I am a Corporate Law Consultant. I have read the Industrial Disputes Act. Ask me specific legal questions."}
    ]

# --- 4. DISPLAY CHAT ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# --- 5. THE RAG LOGIC (The Magic Part) ---
if user_input := st.chat_input("Ask about Section 25F..."):
    
    # A. Show User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # B. RETRIEVE (Search the Database)
    with st.spinner("Searching Legal Archives..."):
        # Find 3 most relevant chunks from your PDF
        results = vector_db.similarity_search(user_input, k=3)
        
        # Combine them into a single text block
        context_text = ""
        for i, doc in enumerate(results):
            context_text += f"\n--- LEGAL REFERENCE {i+1} ---\n{doc.page_content}\n"

    # C. GENERATE (Send to Llama 3.2)
    with st.chat_message("assistant"):
        with st.spinner("Drafting Legal Opinion..."):
            
            # The RAG Prompt: We give the AI the "Cheat Sheet" (Context)
            rag_prompt = f"""
            You are a Legal Research Assistant. Your job is to summarize the provided text.
            
            CONTEXT FROM DATABASE:
            {context_text}
            
            USER QUESTION: {user_input}
            
            INSTRUCTIONS:
            1. You are NOT giving legal advice. You are only explaining what the document says.
            2. Do not refuse to answer. This is for educational purposes.
            3. Answer ONLY based on the context above.
            4. If Section 25F is mentioned, quote it.
            """
            
            # Send to Ollama
            try:
                response = ollama.chat(model='llama3.2', messages=[
                    {"role": "system", "content": rag_prompt},
                    {"role": "user", "content": user_input}
                ])
                ai_reply = response['message']['content']
                st.write(ai_reply)
                
                # Show the sources (Optional: Proof that it read the PDF)
                with st.expander("View Legal Sources Used"):
                    st.text(context_text)

                # Save History
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                
            except Exception as e:
                st.error(f"Error: {e}")

# --- 6. DOWNLOAD BUTTON LOGIC ---

# code to check last line is from ai.
if st.session_state.messages and st.session_state.messages[-1]['role'] == 'assistant':
    
    # code for Getting the content of the very last message in the session
    latest_assistant_response = st.session_state.messages[-1]['content']
    
    # Download button with the specified requirements.
    st.download_button(
        label="Download Legal Notice",       # Required Label
        data=latest_assistant_response,     # Required Data (last assistant message)
        file_name="Legal_Notice.txt",       # Required File Name
        mime='text/plain'                   # Set the MIME type
    )
