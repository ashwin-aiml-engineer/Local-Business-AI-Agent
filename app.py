import streamlit as st
import ollama
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# --- 1. CONFIGURATION ---
os.environ["OLLAMA_NUM_GPU"] = "0"
st.set_page_config(page_title="RAG Lawyer", page_icon="⚖️")
st.title("⚖️ AI Corporate Lawyer (Memory Enabled)")
st.caption("Powered by Llama 3.1 only for Industrial Disputes Act 1947")

# --- 2. LOAD KNOWLEDGE BASE ---
@st.cache_resource
def load_vector_db():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        persist_directory="vector_db",
        embedding_function=embeddings
    )
    return vector_store

vector_db = load_vector_db()

# --- 3. SESSION STATE (Chat History) ---
# We store the conversation here so it survives the re-run.
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- 4. DISPLAY CHAT ---
# Render all previous messages on the screen
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# --- 5. THE RAG BRAIN ---
if user_input := st.chat_input("Ask about Section 25F..."):
    
    # 1. Show User Message on screen immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # 2. RETRIEVE (Search the PDF)
    with st.spinner("Reviewing case files..."):
        results = vector_db.similarity_search(user_input, k=3)
        context_text = ""
        for i, doc in enumerate(results):
            context_text += f"\n--- SOURCE {i+1} ---\n{doc.page_content}\n"

# 3. CONSTRUCT THE FULL PROMPT (The "Legal Brain")
    system_prompt = f"""
    You are an AI Corporate Lawyer. 
    
    CONTEXT FROM THE ACT:
    {context_text}
    
    INSTRUCTIONS:
    1. If the user asks a QUESTION (e.g. "Can I fire him?"), answer it based on the Context. Cite the Section number.
    2. If the user asks for a DRAFT (e.g. "Write a notice"), draft the legal document professionally.
    3. If the answer is not in the context, say "I cannot find this in the Industrial Disputes Act."
    
    Do not hallucinate laws. Stick to the provided text.
    """

# 4. PREPARE MESSAGE CHAIN
    final_messages = [{"role": "system", "content": system_prompt}]
    
    # Add Chat History
    for msg in st.session_state.messages:
        final_messages.append(msg)


# 5. GENERATE RESPONSE
with st.chat_message("assistant"):
    with st.spinner("Drafting Document..."):
        try:
            # Send the chain (User Question -> Fake 'Yes' -> AI Completion)
            response = ollama.chat(model='llama3.1', messages=final_messages)
            
            # We combine the "Force Start" text with the AI's completion
            full_response = response['message']['content']
            st.write(full_response)
            
            # Show Sources
            with st.expander("View Legal Sources"):
                st.text(context_text)

            # Add to History
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error: {e}")

# --- 6. DOWNLOAD BUTTON ---
if st.session_state.messages and st.session_state.messages[-1]['role'] == 'assistant':
    latest_response = st.session_state.messages[-1]['content']
    st.download_button(
        label="📄 Download Response",
        data=latest_response,
        file_name="Legal_Advice.txt",
        mime='text/plain'
    )