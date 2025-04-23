import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import time
import pickle
import os
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# For BERT model
try:
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification, AdamW
    from transformers import get_linear_schedule_with_warmup
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

# Download NLTK resources
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    st.warning("NLTK resources couldn't be downloaded. Some functionality might be limited.")

# Page Configuration
st.set_page_config(
    page_title="Toxic Comment Detection", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application Constants
MODEL_PATH = "models/"
DATA_PATH = "data/"
TOXIC_LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MAX_FEATURES = 100000  # Maximum number of features for TF-IDF
MAX_SEQUENCE_LENGTH = 200  # For deep learning models
EMBEDDING_DIM = 100  # For word embeddings

# Ensure directories exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# Text preprocessing functions
@st.cache_data
def clean_text(text):
    """Clean and preprocess text data"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

@st.cache_data
def preprocess_text(texts):
    """Apply preprocessing to a list of texts"""
    return [clean_text(text) for text in texts]

@st.cache_data
def load_data(file_path=None):
    """
    Load and preprocess the toxic comment dataset
    """
    try:
        if file_path:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            else:
                data = pd.read_csv(file_path + "/train.csv")
        else:
            # Use sample data if no file is provided
            data = pd.read_csv("https://raw.githubusercontent.com/AI4Bharat/indicnlp_corpus/master/samples/classification/train.csv")
        
        # Display basic information
        st.write(f"Dataset loaded successfully! Shape: {data.shape}")
        
        # Preprocess the comments
        data['cleaned_comment'] = preprocess_text(data['comment_text'])
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_tfidf_vectors(texts, max_features=MAX_FEATURES):
    """Create TF-IDF vectors from text data"""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=5,
        max_df=0.7
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

@st.cache_resource
def create_tokenizer(texts, num_words=MAX_FEATURES):
    """Create and fit a tokenizer for deep learning models"""
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def vectorize_sequences(texts, tokenizer):
    """Convert texts to sequences and pad them"""
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

@st.cache_resource
def train_baseline_model(X_train, y_train, model_type="logistic"):
    """Train a baseline model (Logistic Regression or Naive Bayes)"""
    if model_type == "logistic":
        model = MultiOutputClassifier(LogisticRegression(max_iter=500, C=5, class_weight='balanced'))
    else:  # Naive Bayes
        model = MultiOutputClassifier(MultinomialNB(alpha=0.1))
    
    model.fit(X_train, y_train)
    return model

@st.cache_resource
def build_lstm_model(vocab_size, embedding_dim=EMBEDDING_DIM):
    """Build and compile an LSTM model"""
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=MAX_SEQUENCE_LENGTH))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.5))
    model.add(Dense(len(TOXIC_LABELS), activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

@st.cache_resource
def initialize_bert_model():
    """Initialize a pre-trained BERT model for fine-tuning"""
    if BERT_AVAILABLE:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=len(TOXIC_LABELS),
            problem_type="multi_label_classification"
        )
        return tokenizer, model
    return None, None

def save_model(model, model_path, model_type):
    """Save trained model to disk"""
    if model_type in ["logistic", "naive_bayes"]:
        with open(f"{model_path}{model_type}_model.pkl", 'wb') as f:
            pickle.dump(model, f)
    elif model_type == "lstm":
        model.save(f"{model_path}lstm_model.h5")
    elif model_type == "bert":
        model.save_pretrained(f"{model_path}bert_model")
    
    st.success(f"{model_type.capitalize()} model saved successfully!")

def load_model_from_disk(model_path, model_type):
    """Load a trained model from disk"""
    try:
        if model_type in ["logistic", "naive_bayes"]:
            with open(f"{model_path}{model_type}_model.pkl", 'rb') as f:
                return pickle.load(f)
        elif model_type == "lstm":
            return load_model(f"{model_path}lstm_model.h5")
        elif model_type == "bert" and BERT_AVAILABLE:
            return BertForSequenceClassification.from_pretrained(f"{model_path}bert_model")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_toxicity(text, model, vectorizer=None, tokenizer=None, model_type="logistic"):
    """Make toxicity predictions on new text"""
    cleaned_text = clean_text(text)
    
    if model_type in ["logistic", "naive_bayes"]:
        X = vectorizer.transform([cleaned_text])
        predictions = model.predict(X)[0]
    
    elif model_type == "lstm":
        sequence = vectorize_sequences([cleaned_text], tokenizer)
        predictions = model.predict(sequence)[0]
    
    elif model_type == "bert" and BERT_AVAILABLE:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        inputs = bert_tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQUENCE_LENGTH)
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.sigmoid(outputs.logits).numpy()[0]
    
    else:
        return None
    
    return {label: float(pred) for label, pred in zip(TOXIC_LABELS, predictions)}

def plot_class_distribution(data):
    """Plot the distribution of classes in the dataset"""
    class_counts = data[TOXIC_LABELS].sum().sort_values(ascending=False)
    
    fig = px.bar(
        x=class_counts.index, 
        y=class_counts.values,
        labels={'x': 'Toxicity Category', 'y': 'Count'},
        title='Distribution of Toxicity Categories',
        color=class_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title="Toxicity Category",
        yaxis_title="Count",
        coloraxis_showscale=False
    )
    
    return fig

def plot_wordcloud(data, toxic=True):
    """Create a wordcloud from toxic or non-toxic comments"""
    if toxic:
        # Get comments that have at least one toxic label
        mask = data[TOXIC_LABELS].sum(axis=1) > 0
        title = "Word Cloud of Toxic Comments"
    else:
        # Get comments that have no toxic labels
        mask = data[TOXIC_LABELS].sum(axis=1) == 0
        title = "Word Cloud of Non-Toxic Comments"
    
    filtered_comments = ' '.join(data[mask]['cleaned_comment'])
    
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=200,
        contour_width=3,
        contour_color='steelblue'
    ).generate(filtered_comments)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud)
    ax.set_title(title)
    ax.axis('off')
    
    return fig

def plot_roc_curves(y_test, y_pred_proba):
    """Plot ROC curves for each toxic category"""
    fig = go.Figure()
    
    for i, label in enumerate(TOXIC_LABELS):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
        auc_score = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'{label} (AUC = {auc_score:.3f})',
            mode='lines'
        ))
    
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    
    fig.update_layout(
        title='ROC Curves for Toxicity Categories',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.7, y=0.1),
        width=800, height=600
    )
    
    return fig

def plot_precision_recall_curves(y_test, y_pred_proba):
    """Plot Precision-Recall curves for each toxic category"""
    fig = go.Figure()
    
    for i, label in enumerate(TOXIC_LABELS):
        precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            name=f'{label}',
            mode='lines'
        ))
    
    fig.update_layout(
        title='Precision-Recall Curves for Toxicity Categories',
        xaxis_title='Recall',
        yaxis_title='Precision',
        legend=dict(x=0.7, y=0.1),
        width=800, height=600
    )
    
    return fig

def plot_confusion_matrices(y_test, y_pred):
    """Plot confusion matrices for each toxic category"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (ax, label) in enumerate(zip(axes, TOXIC_LABELS)):
        cm = confusion_matrix(y_test[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix: {label.replace("_", " ").title()}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
    
    plt.tight_layout()
    return fig

def main():
    st.title("🛡️ Toxic Comment Detection System")
    st.markdown("""
    This application helps detect toxic comments in online conversations using machine learning.
    You can train different models, evaluate their performance, and analyze new comments in real-time.
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Training", "Comment Analysis"])
    
    # Only show these options on the Home page
    if page == "Home":
        st.header("Welcome to the Toxic Comment Detection System")
        st.markdown("""
        ## About This Project
        
        This application helps identify toxic comments in online discussions using Natural Language Processing and Machine Learning techniques.
        
        ### Features:
        - **Data Exploration**: Analyze the toxic comment dataset with visualizations
        - **Model Training**: Train and evaluate different ML models for toxicity detection
        - **Comment Analysis**: Test the trained models on new comments in real-time
        
        ### Getting Started:
        1. Upload your dataset or use our sample data
        2. Explore data characteristics and patterns
        3. Train one or more models
        4. Analyze new comments for toxicity
        
        ### Supported Models:
        - Logistic Regression (Baseline)
        - Naive Bayes (Baseline)
        - Bidirectional LSTM (Deep Learning)
        - BERT (Transformer-based, if available)
        """)
        
        # Show sample comment analysis on home page
        st.header("Try a Quick Demo")
        demo_comment = st.text_area(
            "Enter a comment to analyze:",
            "This is a test comment. Let's see if it's toxic or not.",
            height=100
        )
        
        if st.button("Analyze Comment"):
            # For demo purposes use a pre-trained model or simple heuristics
            st.info("This is a demo prediction. For accurate results, train a model first.")
            
            # Simple demo prediction (replace with actual model if available)
            toxic_words = ['hate', 'idiot', 'stupid', 'dumb', 'fool', 'kill', 'die']
            has_toxic_words = any(word in demo_comment.lower() for word in toxic_words)
            
            if has_toxic_words:
                st.error("⚠️ This comment may contain toxic content.")
            else:
                st.success("✅ This comment appears to be non-toxic.")
    
    # Data Exploration Page
    elif page == "Data Exploration":
        st.header("Data Exploration")
        
        # Data upload options
        st.subheader("Upload Dataset")
        upload_method = st.radio(
            "Select data source:",
            ["Upload CSV", "Use Kaggle dataset link", "Use sample data"]
        )
        
        data = None
        
        if upload_method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload toxic comments dataset (CSV)", type=["csv"])
            if uploaded_file is not None:
                # Save uploaded file
                with open(os.path.join(DATA_PATH, "uploaded_data.csv"), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                data = load_data(os.path.join(DATA_PATH, "uploaded_data.csv"))
        
        elif upload_method == "Use Kaggle dataset link":
            kaggle_link = st.text_input(
                "Enter Kaggle dataset link or path:",
                "https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data"
            )
            if st.button("Load Data") and kaggle_link:
                st.info("For Kaggle datasets, you should download the data manually and upload the CSV file.")
                data = load_data()  # Use sample data for demonstration
        
        elif upload_method == "Use sample data":
            if st.button("Load Sample Data"):
                data = load_data()
        
        # Display data exploration if data is loaded
        if data is not None:
            st.write(data.head())
            
            st.subheader("Dataset Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Comments", f"{len(data):,}")
                
                toxic_count = (data[TOXIC_LABELS].sum(axis=1) > 0).sum()
                st.metric("Toxic Comments", f"{toxic_count:,} ({toxic_count/len(data):.1%})")
            
            with col2:
                avg_length = data['cleaned_comment'].str.len().mean()
                st.metric("Average Comment Length", f"{avg_length:.1f} characters")
                
                num_null = data['comment_text'].isnull().sum()
                st.metric("Missing Values", f"{num_null:,}")
            
            st.subheader("Distribution of Toxicity Categories")
            distribution_plot = plot_class_distribution(data)
            st.plotly_chart(distribution_plot, use_container_width=True)
            
            st.subheader("Word Clouds")
            col1, col2 = st.columns(2)
            with col1:
                toxic_wordcloud = plot_wordcloud(data, toxic=True)
                st.pyplot(toxic_wordcloud)
            
            with col2:
                non_toxic_wordcloud = plot_wordcloud(data, toxic=False)
                st.pyplot(non_toxic_wordcloud)
            
            st.subheader("Sample Comments")
            st.markdown("### Toxic Comments Sample")
            toxic_samples = data[data[TOXIC_LABELS].sum(axis=1) > 0].sample(5)
            for idx, row in toxic_samples.iterrows():
                categories = ', '.join([label for label, val in row[TOXIC_LABELS].items() if val > 0])
                st.text_area(f"ID {idx}: {categories}", row['comment_text'], height=100)
            
            st.markdown("### Non-Toxic Comments Sample")
            non_toxic_samples = data[data[TOXIC_LABELS].sum(axis=1) == 0].sample(5)
            for idx, row in non_toxic_samples.iterrows():
                st.text_area(f"ID {idx}: Clean", row['comment_text'], height=100)
    
    # Model Training Page
    elif page == "Model Training":
        st.header("Model Training & Evaluation")
        
        # Load data first
        st.subheader("1. Load Dataset")
        upload_method = st.radio(
            "Select data source:",
            ["Upload CSV", "Use sample data"],
            key="train_data_source"
        )
        
        data = None
        
        if upload_method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload toxic comments dataset (CSV)", type=["csv"], key="train_upload")
            if uploaded_file is not None:
                with open(os.path.join(DATA_PATH, "train_data.csv"), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                data = load_data(os.path.join(DATA_PATH, "train_data.csv"))
        else:
            if st.button("Load Sample Data", key="train_sample"):
                data = load_data()
        
        if data is not None:
            st.success(f"Dataset loaded with {len(data)} comments")
            
            # Model configuration
            st.subheader("2. Configure Model")
            
            model_type = st.selectbox(
                "Select model type:",
                ["logistic", "naive_bayes", "lstm"]
            )
            
            if BERT_AVAILABLE:
                if model_type == "bert":
                    st.info("BERT model requires significant computing resources")
            else:
                st.warning("BERT model is not available due to missing dependencies")
            
            # Handle class imbalance
            st.markdown("#### Class Imbalance Handling")
            imbalance_method = st.selectbox(
                "Method to handle class imbalance:",
                ["None", "Undersampling", "Oversampling (SMOTE)", "Class weights"]
            )
            
            # Training settings
            st.markdown("#### Training Settings")
            test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
            
            if model_type == "lstm":
                batch_size = st.number_input("Batch size", 16, 256, 64, 16)
                epochs = st.number_input("Number of epochs", 1, 20, 5, 1)
            
            # Start training
            if st.button("Train Model"):
                with st.spinner("Preprocessing data..."):
                    # Split data
                    X = data['cleaned_comment']
                    y = data[TOXIC_LABELS].values
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=data['toxic']
                    )
                    
                    # Handle class imbalance
                    if imbalance_method == "Undersampling":
                        rus = RandomUnderSampler(random_state=42)
                        # We need to reshape for undersampling
                        indices_train = np.arange(len(X_train)).reshape(-1, 1)
                        indices_resampled, y_train = rus.fit_resample(indices_train, y_train)
                        X_train = X_train.iloc[indices_resampled.flatten()]
                        st.info(f"After undersampling: {len(X_train)} training samples")
                    
                    # Process text based on model type
                    if model_type in ["logistic", "naive_bayes"]:
                        # Create TF-IDF vectors
                        X_train_vec, vectorizer = create_tfidf_vectors(X_train)
                        X_test_vec, _ = create_tfidf_vectors(X_test, max_features=MAX_FEATURES)
                    
                    elif model_type == "lstm":
                        # Create sequences
                        tokenizer = create_tokenizer(X_train)
                        X_train_seq = vectorize_sequences(X_train, tokenizer)
                        X_test_seq = vectorize_sequences(X_test, tokenizer)
                
                with st.spinner(f"Training {model_type} model..."):
                    if model_type in ["logistic", "naive_bayes"]:
                        model = train_baseline_model(X_train_vec, y_train, model_type)
                        y_pred = model.predict(X_test_vec)
                        y_pred_proba = model.predict_proba(X_test_vec)
                        
                        # Extract probabilities
                        y_pred_proba_array = np.zeros((len(y_pred), len(TOXIC_LABELS)))
                        for i, estimator in enumerate(model.estimators_):
                            y_pred_proba_array[:, i] = estimator.predict_proba(X_test_vec)[:, 1]
                        
                    elif model_type == "lstm":
                        vocab_size = len(tokenizer.word_index) + 1
                        model = build_lstm_model(vocab_size)
                        
                        # Train with early stopping
                        early_stopping = EarlyStopping(
                            monitor='val_loss',
                            patience=2,
                            restore_best_weights=True
                        )
                        
                        history = model.fit(
                            X_train_seq, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.1,
                            callbacks=[early_stopping],
                            verbose=0
                        )
                        
                        # Make predictions
                        y_pred_proba_array = model.predict(X_test_seq)
                        y_pred = (y_pred_proba_array > 0.5).astype(int)
                
                # Save the trained model and necessary components
                if model_type in ["logistic", "naive_bayes"]:
                    with open(f"{MODEL_PATH}{model_type}_vectorizer.pkl", 'wb') as f:
                        pickle.dump(vectorizer, f)
                
                elif model_type == "lstm":
                    with open(f"{MODEL_PATH}lstm_tokenizer.pkl", 'wb') as f:
                        pickle.dump(tokenizer, f)
                
                save_model(model, MODEL_PATH, model_type)
                
                # Display evaluation results
                st.subheader("3. Model Evaluation")
                
                # Classification report
                st.markdown("#### Classification Report")
                report = classification_report(y_test, y_pred, target_names=TOXIC_LABELS, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
                
                # ROC curves
                st.markdown("#### ROC Curves")
                roc_fig = plot_roc_curves(y_test, y_pred_proba_array)
                st.plotly_chart(roc_fig)
                
                # Precision-Recall curves
                st.markdown("#### Precision-Recall Curves")
                pr_fig = plot_precision_recall_curves(y_test, y_pred_proba_array)
                st.plotly_chart(pr_fig)
                
                # Confusion matrices
                st.markdown("#### Confusion Matrices")
                cm_fig = plot_confusion_matrices(y_test, y_pred)
                st.pyplot(cm_fig)
                
                st.success("✅ Model training and evaluation completed!")
    
    # Comment Analysis Page
    elif page == "Comment Analysis":
        st.header("Comment Analysis")
        
        # Model selection
        st.subheader("1. Select Model")
        available_models = []
        
        if os.path.exists(f"{MODEL_PATH}logistic_model.pkl"):
            available_models.append("logistic")
        if os.path.exists(f"{MODEL_PATH}naive_bayes_model.pkl"):
            available_models.append("naive_bayes")
        if os.path.exists(f"{MODEL_PATH}lstm_model.h5"):
            available_models.append("lstm")
        if os.path.exists(f"{MODEL_PATH}bert_model") and BERT_AVAILABLE:
            available_models.append("bert")
        
        if not available_models:
            st.warning("No trained models found. Please go to the Model Training page first.")
        else:
            model_to_use = st.selectbox("Select a model to use:", available_models)
            
            # Load model and required components
            model = load_model_from_disk(MODEL_PATH, model_to_use)
            
            vectorizer = None
            tokenizer = None
            
            if model_to_use in ["logistic", "naive_bayes"]:
                with open(f"{MODEL_PATH}{model_to_use}_vectorizer.pkl", 'rb') as f:
                    vectorizer = pickle.load(f)
            elif model_to_use == "lstm":
                with open(f"{MODEL_PATH}lstm_tokenizer.pkl", 'rb') as f:
                    tokenizer = pickle.load(f)
            
            # Input options
            st.subheader("2. Enter Text to Analyze")
            
            analysis_option = st.radio(
                "Choose input method:",
                ["Enter text", "Upload file", "Use example comments"]
            )
            
            if analysis_option == "Enter text":
                comment = st.text_area(
                    "Enter a comment to analyze:",
                    "This is a test comment. Let's analyze it for toxicity.",
                    height=150
                )
                
                if st.button("Analyze"):
                    with st.spinner("Analyzing comment..."):
                        predictions = predict_toxicity(
                            comment, model, vectorizer, tokenizer, model_to_use
                        )
                        
                        st.subheader("Analysis Results")
                        
                        # Create a gauge chart for each category
                        col1, col2, col3 = st.columns(3)
                        cols = [col1, col2, col3, col1, col2, col3]
                        
                        any_toxic = False
                        for i, (label, score) in enumerate(predictions.items()):
                            with cols[i]:
                                # Create gauge
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=score * 100,
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={'text': label.replace('_', ' ').title()},
                                    gauge={
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': "darkblue"},
                                        'steps': [
                                            {'range': [0, 50], 'color': "lightgreen"},
                                            {'range': [50, 75], 'color': "orange"},
                                            {'range': [75, 100], 'color': "red"},
                                        ],
                                       'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 50
                                        }
                                    }
                                ))
                                
                                fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
                                st.plotly_chart(fig, use_container_width=True)
                                
                                if score > 0.5:
                                    any_toxic = True
                        
                        # Overall assessment
                        st.subheader("Overall Assessment")
                        if any_toxic:
                            st.error("⚠️ This comment contains potentially toxic content.")
                        else:
                            st.success("✅ This comment appears to be non-toxic.")
                        
                        # Highlight potentially problematic words
                        st.subheader("Text Analysis")
                        
                        # Simple highlighting of potentially toxic words (could be improved with more sophisticated methods)
                        toxic_words = {
                            'toxic': ['hate', 'idiot', 'stupid', 'fuck', 'shit'],
                            'severe_toxic': ['fucking', 'kill', 'die', 'death'],
                            'obscene': ['ass', 'bitch', 'fuck', 'shit', 'cunt'],
                            'threat': ['kill', 'die', 'threat', 'murder', 'attack'],
                            'insult': ['idiot', 'stupid', 'dumb', 'fool', 'loser'],
                            'identity_hate': ['racist', 'sexist', 'gay', 'black', 'white', 'jew', 'muslim']
                        }
                        
                        words = comment.lower().split()
                        highlighted_text = comment
                        
                        for category, word_list in toxic_words.items():
                            if predictions[category] > 0.5:
                                for word in word_list:
                                    if word in comment.lower():
                                        highlighted_text = highlighted_text.replace(
                                            word, f"<span style='background-color: yellow'>{word}</span>"
                                        )
                        
                        st.markdown(f"<div style='background-color: #f9f9f9; padding: 15px; border-radius: 5px;'>{highlighted_text}</div>", unsafe_allow_html=True)
            
            elif analysis_option == "Upload file":
                uploaded_file = st.file_uploader("Upload a text file with comments", type=["txt"])
                
                if uploaded_file is not None:
                    content = uploaded_file.getvalue().decode("utf-8")
                    comments = content.split("\n")
                    
                    if st.button("Analyze File"):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, comment in enumerate(comments):
                            if comment.strip():
                                predictions = predict_toxicity(
                                    comment, model, vectorizer, tokenizer, model_to_use
                                )
                                
                                is_toxic = any(score > 0.5 for score in predictions.values())
                                results.append({
                                    "Comment": comment,
                                    "Is Toxic": is_toxic,
                                    **predictions
                                })
                            
                            progress_bar.progress((i + 1) / len(comments))
                        
                        results_df = pd.DataFrame(results)
                        
                        st.subheader("Analysis Results")
                        st.write(f"Total comments analyzed: {len(results)}")
                        st.write(f"Toxic comments found: {results_df['Is Toxic'].sum()}")
                        
                        st.dataframe(results_df)
                        
                        # Plot distribution of toxic categories
                        fig = px.bar(
                            x=TOXIC_LABELS,
                            y=[(results_df[col] > 0.5).sum() for col in TOXIC_LABELS],
                            labels={'x': 'Category', 'y': 'Count'},
                            title='Distribution of Detected Toxic Categories'
                        )
                        st.plotly_chart(fig)
            
            elif analysis_option == "Use example comments":
                examples = [
                    "I really enjoyed reading this article, it was very informative.",
                    "You're an idiot if you believe that nonsense.",
                    "I hope you have a wonderful day!",
                    "Go kill yourself, nobody likes you anyway.",
                    "I disagree with your opinion, but I respect your right to express it."
                ]
                
                selected_example = st.selectbox("Select an example comment:", examples)
                
                if st.button("Analyze"):
                    with st.spinner("Analyzing comment..."):
                        predictions = predict_toxicity(
                            selected_example, model, vectorizer, tokenizer, model_type=model_to_use
                        )
                        
                        st.subheader("Analysis Results")
                        
                        # Display as horizontal bar chart
                        fig = px.bar(
                            x=[predictions[label] for label in TOXIC_LABELS],
                            y=[label.replace('_', ' ').title() for label in TOXIC_LABELS],
                            orientation='h',
                            labels={'x': 'Probability', 'y': 'Category'},
                            title='Toxicity Probabilities',
                            range_x=[0, 1]
                        )
                        
                        # Add vertical line at threshold
                        fig.add_shape(
                            type="line",
                            x0=0.5, y0=-0.5,
                            x1=0.5, y1=len(TOXIC_LABELS)-0.5,
                            line=dict(color="red", width=2, dash="dash")
                        )
                        
                        st.plotly_chart(fig)
                        
                        # Overall assessment
                        if any(score > 0.5 for score in predictions.values()):
                            categories = [label.replace('_', ' ').title() for label, score in predictions.items() if score > 0.5]
                            st.error(f"⚠️ This comment may contain {', '.join(categories)} content.")
                        else:
                            st.success("✅ This comment appears to be non-toxic.")

if __name__ == "__main__":
    main()
