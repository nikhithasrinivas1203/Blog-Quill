import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the LLaMA 2 model and tokenizer
@st.cache_resource
def load_model():
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with the correct model path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit app
st.title("Blog Generator using LLaMA 2")

# Input from user
prompt = st.text_area("Enter your blog topic or prompt:", height=100)

# Generate blog content
if st.button("Generate Blog"):
    if prompt:
        with st.spinner("Generating blog content..."):
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=500, num_return_sequences=1)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write(generated_text)
    else:
        st.warning("Please enter a prompt to generate the blog.")