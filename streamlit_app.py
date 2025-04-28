import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Access the Hugging Face token from Streamlit secrets
HF_TOKEN = st.secrets["hf_token"]

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define toxicity labels (must match training setup)
TOXICITY_LABELS = [
    'toxic', 
    'severe_toxic', 
    'obscene', 
    'threat', 
    'insult', 
    'identity_hate'
]

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
        num_labels=len(TOXICITY_LABELS))
    
    # Load the saved state_dict into the model
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Load the model
model = load_model()

# Streamlit UI
st.title("Toxic Comment Detector")
st.write("Analyze comments for multiple types of toxicity")

# Text input from user
user_input = st.text_area("Enter your comment here:", height=150)

# Add threshold slider
threshold = st.slider(
    "Classification Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.01,
    help="Adjust the sensitivity for toxicity detection"
)

if st.button("Analyze Comment"):
    if user_input:
        # Preprocess the input
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.sigmoid(outputs.logits).squeeze().numpy()
        
        # Create results dictionary
        results = {label: float(prob) for label, prob in zip(TOXICITY_LABELS, probabilities)}
        
        # Display results
        st.subheader("Toxicity Analysis:")
        
        # Show probability bars
        for label, prob in results.items():
            st.progress(
                prob, 
                text=f"{label.replace('_', ' ').title()}: {prob:.4f}"
            )
        
        # Determine if any toxicity exceeds threshold
        toxic_detected = any(prob > threshold for prob in results.values())
        
        # Show overall conclusion
        st.subheader("Overall Conclusion:")
        if toxic_detected:
            toxic_labels = [label for label, prob in results.items() if prob > threshold]
            st.error(f"⚠️ Toxic content detected ({', '.join(toxic_labels)})")
        else:
            st.success("✅ No toxic content detected")
            
        # Optional: Add a bar chart visualization
        st.bar_chart(results)
        
    else:
        st.warning("Please enter a comment to analyze")

# Optional: Add explanation expander
with st.expander("How this works:"):
    st.markdown("""
    This toxicity detector analyzes text for 6 types of harmful content:
    - **Toxic**: Generally rude, disrespectful, or unreasonable language
    - **Severe Toxic**: Extremely hateful, aggressive, or harassing content
    - **Obscene**: Vulgar or lewd language including profanity
    - **Threat**: Intent to inflict physical or emotional harm
    - **Insult**: Language intended to demean or offend
    - **Identity Hate**: Prejudiced attacks on protected characteristics
    
    The model uses BERT-based machine learning trained on Wikipedia comments.
    Adjust the threshold slider to control detection sensitivity.
    """)
