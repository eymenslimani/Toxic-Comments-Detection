import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer
model_path = 'models/best_model.pt'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # CPU for portability
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define labels
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Streamlit UI
st.title("Toxic Comment Classifier")
st.markdown("Enter a comment below to analyze its toxicity levels.")

# Input area
comment = st.text_area("Comment", height=150, placeholder="Type your comment here...")

if st.button("Classify"):
    if comment:
        # Tokenize input
        encoding = tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Display results
        st.subheader("Toxicity Analysis:")
        for label, prob in zip(labels, probs):
            st.write(f"{label.capitalize()}: {prob:.2%}")
            st.progress(prob)
        
        # Toxicity warning
        if any(prob > 0.5 for prob in probs):
            st.error("⚠️ This comment contains toxic content!")
        else:
            st.success("✅ This comment appears safe.")
    else:
        st.warning("Please enter a comment to classify.")

# Custom styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #ced4da;
    }
    </style>
""", unsafe_allow_html=True)
