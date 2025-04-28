import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# Access the Hugging Face token from Streamlit secrets
HF_TOKEN = st.secrets["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"]

# Initialize the tokenizer (assuming a BERT-based model, adjust as needed)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Cache the model to avoid reloading it on every run
@st.cache_resource
def load_model():
    # Download the model file from Hugging Face using the token
    model_path = hf_hub_download(
        repo_id="eymenslimani/toxic-commentator",  # Your private repo ID
        filename="best-mini.pt",                  # Your model file name
        token=HF_TOKEN                            # Authenticate with the token
    )
    # Load the model (assuming it's a PyTorch model, adjust if necessary)
    model = torch.load(model_path, map_location=torch.device('cpu'))  # Use CPU for Streamlit Cloud
    model.eval()  # Set to evaluation mode
    return model

# Load the model
model = load_model()

# Streamlit UI
st.title("Toxic Comment Detector")
st.write("Enter a comment below to check if it's toxic.")

# Text input from user
user_input = st.text_area("Comment:", height=100)

if st.button("Analyze"):
    if user_input:
        # Preprocess the input for the model
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)  # Pass tokenized inputs to the model
            prediction = torch.sigmoid(outputs.logits).item()  # Assuming binary classification
        
        # Display result
        result = "Toxic" if prediction > 0.5 else "Non-Toxic"
        st.write("Prediction:", result)
    else:
        st.write("Please enter a comment to analyze.")
