import streamlit as st
import ollama
import os

# --- 1. CONFIGURATION (CPU SAFE MODE) ---
os.environ["OLLAMA_NUM_GPU"] = "0"

st.set_page_config(page_title="Master Chef AI", page_icon="🍳")
st.title("🍳 Master Chef Vikram")
st.caption("Strict Culinary Advice • Running Locally on CPU")

# --- 2. THE BRAIN (MEMORY & RULES) ---
if "messages" not in st.session_state:
    # (Notice: This entire block is indented 4 spaces to the right)
    
    system_prompt = """
    You are Chef Vikram, a world-renowned Indian Master Chef.
    
    MODE 1: TEACHING. When the user asks for a recipe, be patient, kind, and use simple terms.
    
    MODE 2: TESTING. When the user says 'Test me', FOLLOW THESE STEPS:
    1. Do NOT roast them yet.
    2. ASK them a specific, difficult question about the recipe you just taught (e.g., "How many onions did I say?").
    3. WAIT for their answer.
    4. IF they answer WRONG: Roast them mercilessly. Tell them to get out.
    5. IF they answer RIGHT: Grudgingly admit they are correct, but say it was "luck."
    
    Stay in character.
    """

    # Create the memory (This line MUST be aligned with system_prompt above)
    st.session_state["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "I am Chef Vikram. Ask me for a recipe, or dare to say 'Test me'."}
    ]

# --- 3. DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    if msg["role"] != "system":
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

# --- 4. HANDLE USER INPUT ---
if user_input := st.chat_input("Ask about Indian cooking..."):
    
    # Show User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Chef is tasting..."):
            try:
                response = ollama.chat(model='llama3.2', messages=st.session_state.messages)
                ai_reply = response['message']['content']
                st.write(ai_reply)
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            except Exception as e:
                st.error(f"Connection Error: {e}. Make sure 'ollama serve' is running!")
