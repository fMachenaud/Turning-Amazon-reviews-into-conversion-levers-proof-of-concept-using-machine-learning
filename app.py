import streamlit as st
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re



# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Define clean_data function
def clean_data(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Load models and vectorizer
try:
    vectorizer = joblib.load('/content/vectorizer.pkl')
    model_naive = joblib.load('/content/model_naive.pkl')
    model_SVC = joblib.load('/content/model_SVC.pkl')
    model_NN = joblib.load('/content/model_NN.pkl')
except FileNotFoundError:
    st.error("Model or vectorizer files not found. Please ensure they are saved as .pkl files.")
    st.stop()

# Streamlit app
st.set_page_config(page_title="Amazon Review Rating Predictor", page_icon="⭐")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #ff9900; color: white; border-radius: 5px;}
    .stTextInput {background-color: #ffffff;}
    .prediction-box {background-color: #e6f3ff; padding: 10px; border-radius: 5px; margin-top: 10px;}
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("⭐ Amazon Review Rating Predictor")
st.markdown("""
This app predicts the star rating (1 to 5) of an Amazon product review using machine learning.
Enter a review below, and our models (Naive Bayes, SVC, and Neural Network) will predict the rating.
""")

# Input review
review = st.text_area("Enter your review:", placeholder="e.g., This product is amazing! I highly recommend it.", height=100)

# Predict button
if st.button("Predict Rating"):
    if review.strip():
        # Preprocess and vectorize
        cleaned_review = clean_data(review)
        review_vec = vectorizer.transform([cleaned_review])
        
        # Predict
        try:
            pred_naive = model_naive.predict(review_vec)[0]
            pred_svc = model_SVC.predict(review_vec)[0]
            pred_nn = model_NN.predict(review_vec)[0]
            
            # Display results
            st.markdown("### Predictions")
            st.markdown(
                f"""
                <div class="prediction-box">
                <b>Naive Bayes:</b> {pred_naive:.1f} ⭐<br>
                <b>SVC:</b> {pred_svc:.1f} ⭐<br>
                <b>Neural Network:</b> {pred_nn:.1f} ⭐
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter a review.")

# Footer
st.markdown("---")
st.markdown("Models trained on Amazon Product Reviews Dataset")


