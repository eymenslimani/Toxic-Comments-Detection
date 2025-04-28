import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Access the Hugging Face token from Streamlit secrets
HF_TOKEN = st.secrets["hf_token"]

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

@st.cache_resource
def load_model():
    # Download the model file
    model_path = hf_hub_download(
        repo_id="eymenslimani/toxic-commentator",
        filename="best-mini.pt",
        token=HF_TOKEN
    )
    
    # Initialize the model architecture
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=6  # Adjust this to match your model's number of classes
    )
    
    # Load the saved state_dict into the model
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # Set to evaluation mode
    model.eval()
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
