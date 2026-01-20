import streamlit as st
from transformers import pipeline

# Page config
st.set_page_config(
    page_title="Gen AI Text Generator",
    layout="centered"
)

st.title("🤖 Gen AI Text Generator")
st.write("Enter a prompt and generate AI-written text using a pretrained language model.")

# Load model (cached)
@st.cache_resource
def load_generator():
    return pipeline(
        "text-generation",
        model="gpt2"
    )

generator = load_generator()

# User input
prompt = st.text_area(
    "Enter your prompt",
    height=200,
    placeholder="Once upon a time in a futuristic world..."
)

max_len = st.slider("Max output length", 50, 200, 100)

# Generate button
if st.button("Generate"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating text..."):
            output = generator(
                prompt,
                max_length=max_len,
                num_return_sequences=1
            )

        st.subheader("✨ Generated Text")
        st.success(output[0]["generated_text"])
