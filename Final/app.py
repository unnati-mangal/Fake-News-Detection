import streamlit as st
import joblib
import re
import string

# Page Configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Text Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load Model and Vectorizer
@st.cache_resource
def load_assets():
    model = joblib.load('best_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False

# UI Layout
st.title("Fake News Detection App")
st.write("Paste an article title or text below to check whether it is authentic or fake.")

if not assets_loaded:
    st.error("Error loading model files (`best_model.pkl` or `tfidf_vectorizer.pkl`). Please ensure model training is complete.")
else:
    user_input = st.text_area("News Content / Article Text:", height=200, placeholder="Type or paste news text here...")
    
    if st.button("Analyze News"):
        if not user_input.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            cleaned_input = clean_text(user_input)
            vectorized_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(vectorized_input)[0]
            
            st.markdown("---")
            if prediction == 1:
                st.success("**Result:** This news article appears to be **REAL (Authentic)**.")
            else:
                st.error("**Result:** This news article appears to be **FAKE**.")

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(vectorized_input)[0]
                confidence = max(probs) * 100
                st.info(f"Model Confidence: **{confidence:.2f}%**")