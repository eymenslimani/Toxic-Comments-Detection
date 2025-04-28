import streamlit as st
import torch
from huggingface_hub import hf_hub_download

# Cache the model to avoid reloading it on every run
@st.cache_resource
def load_model():
    # Download the model file from Hugging Face
    model_path = hf_hub_download(repo_id="eymenslimani/toxic-commentator", filename="best-mini.pt")
    # Load the model (customize this based on how you saved it)
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
        # Preprocess the input (customize this based on your model's requirements)
        # Example: tokenization, converting to tensor, etc.
        # input_tensor = preprocess(user_input)  # Replace with your preprocessing logic
        
        # Dummy preprocessing placeholder (replace this)
        # For example, if your model expects a tensor, you might need a tokenizer
        input_tensor = user_input  # Replace with actual preprocessing
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)  # Replace with your model's inference logic
        
        # Display result (customize based on your model's output)
        # Assuming a binary classification (toxic > 0.5, non-toxic <= 0.5)
        result = "Toxic" if prediction > 0.5 else "Non-Toxic"
        st.write("Prediction:", result)
    else:
        st.write("Please enter a comment to analyze.")
