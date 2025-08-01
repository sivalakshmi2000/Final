import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (only needed the first time)
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = joblib.load("D:/Final Project 1/archive/fake_job_model.pkl")
vectorizer = joblib.load("D:/Final Project 1/archive/tfidf_vectorizer.pkl")

# Text preprocessing function
def preprocess(text):
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()

    # Tokenization
    tokens = text.split()

    # Remove stopwords and lemmatize
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stopwords.words('english')
    ]
    
    return " ".join(clean_tokens)

# Streamlit UI
st.title("🕵️‍♀️ Fake Job Postings Detector")
st.markdown("Enter a job description below to check if it's **fake or real**.")

user_input = st.text_area("📝 Job Description", height=250)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a job description.")
    else:
        with st.spinner("Analyzing..."):
            preprocessed_text = preprocess(user_input)
            vectorized_input = vectorizer.transform([preprocessed_text])
            prediction = model.predict(vectorized_input)[0]
            
            if prediction == 1:
                st.error("🚨 This job posting is predicted to be **FAKE**.")
            else:
                st.success("✅ This job posting is predicted to be **REAL**.")

# Optional: Footer
st.markdown("---")
st.caption("Project: ML-Based Fake Job Postings Detector | Developed by Sivalakshmi")