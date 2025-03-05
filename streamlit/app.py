import streamlit as st
from mlx_lm import load, generate

st.title("Math Assistant build with MLX")
st.markdown("This math assistant is finetuned with 500 examples of math problems and solutions taken from HuggingFace.")

# Load the Hugging Face model and tokenizer
if "model" not in st.session_state:
    st.session_state["model"], st.session_state["tokenizer"] = load("yenchik/mlx-gemma-2-2b-it-math")

model = st.session_state["model"]
tokenizer = st.session_state["tokenizer"]

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare the prompt using the tokenizer chat template (if available)
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt_formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
    else:
        prompt_formatted = prompt

    # Generate response
    response = generate(model, tokenizer, prompt=prompt_formatted, verbose=True)

    # Display assistant's response
    with st.chat_message("assistant"):
        st.markdown(response)

    # Save assistant's response to session state
    st.session_state.messages.append({"role": "assistant", "content": response})
