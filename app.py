import streamlit as st
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import plotly.express as px
import plotly.figure_factory as ff
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix, accuracy_score
import pickle
import os

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

# Load dataset
try:
    data = pd.read_csv('Amazon Product Reviews Dataset.csv')
except FileNotFoundError:
    st.error("Dataset not found. Please ensure 'Amazon Product Reviews Dataset.csv' is available.")
    st.stop()

# Load models, vectorizer, and metrics
try:
    vectorizer = joblib.load('/content/vectorizer.pkl')
    model_naive = joblib.load('/content/model_naive.pkl')
    model_SVC = joblib.load('/content/model_SVC.pkl')
    model_NN = joblib.load('/content/model_NN.pkl')
    metrics = joblib.load("/content/metrics.pkl")
    confusion_matrices = joblib.load("/content/confusion_matrices.pkl")
except FileNotFoundError:
    st.error("Model, vectorizer, or metrics files not found. Please ensure they are saved as .pkl files.")
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
    .sidebar .sidebar-content {background-color: #ffffff;}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Predict Ratings", "Model & Data Insights"])

if page == "Predict Ratings":
    # Title and description
    st.title("⭐ Amazon Review Rating Predictor")
    st.markdown("""
    This app predicts the star rating (1 to 5) of an Amazon product review using machine learning.
    Enter a review below, and our models (Naive Bayes, SVC, Neural Network) will predict the rating.
    """)

    # Input review
    review = st.text_area("Enter your review:", placeholder="e.g., This product is amazing! I highly recommend it.", height=100)

    # Predict button
    if st.button("Predict Rating"):
        if review.strip():
            # Preprocess and vectorize
            cleaned_review = clean_data(review)
            review_vec = vectorizer.transform([cleaned_review])

            # Predict with confidence scores
            try:
                pred_naive = model_naive.predict(review_vec)[0]
                prob_naive = max(model_naive.predict_proba(review_vec)[0])
                pred_svc = model_SVC.predict(review_vec)[0]
                prob_svc = max(model_SVC.predict_proba(review_vec)[0])
                pred_nn = model_NN.predict(review_vec)[0]
                prob_nn = max(model_NN.predict_proba(review_vec)[0])

                # Display predictions
                st.markdown("### Predictions")
                st.markdown(
                    f"""
                    <div class="prediction-box">
                    <b>Naive Bayes:</b> {pred_naive:.1f} ⭐ (Confidence: {prob_naive:.2%})<br>
                    <b>SVC:</b> {pred_svc:.1f} ⭐ (Confidence: {prob_svc:.2%})<br>
                    <b>Neural Network:</b> {pred_nn:.1f} ⭐ (Confidence: {prob_nn:.2%})
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Sentiment feedback
                avg_rating = (pred_naive + pred_svc + pred_nn) / 3
                if avg_rating >= 4:
                    st.success("Positive review! Great feedback for the product.")
                elif avg_rating <= 2:
                    st.error("Negative review. Consider addressing customer concerns.")
                else:
                    st.warning("Neutral review. Room for improvement.")

                # Predicted rating bar plot
                ratings_df = pd.DataFrame({
                    'Model': ['Naive Bayes', 'SVC', 'Neural Network'],
                    'Rating': [pred_naive, pred_svc, pred_nn]
                })
                fig = px.bar(ratings_df, x='Model', y='Rating', title='Predicted Ratings by Model',
                             range_y=[0, 5], color='Model', text='Rating')
                fig.update_traces(texttemplate='%{text:.1f}', textposition='auto')
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Please enter a review.")

    # Example reviews
    st.markdown("### Test Example Reviews")
    examples = {
        "Positive": "This product is amazing! Works perfectly and highly recommended.",
        "Negative": "Terrible product, broke after one use. Very disappointed.",
        "Neutral": "It’s okay, does the job but nothing special."
    }
    for label, review in examples.items():
        if st.button(f"Predict {label} Review"):
            cleaned_review = clean_data(review)
            review_vec = vectorizer.transform([cleaned_review])

            pred_naive = model_naive.predict(review_vec)[0]
            prob_naive = max(model_naive.predict_proba(review_vec)[0])
            pred_svc = model_SVC.predict(review_vec)[0]
            prob_svc = max(model_SVC.predict_proba(review_vec)[0])
            pred_nn = model_NN.predict(review_vec)[0]
            prob_nn = max(model_NN.predict_proba(review_vec)[0])

            st.markdown(f"**{label} Review**: {review}")
            st.markdown(
                f"""
                <div class="prediction-box">
                <b>Naive Bayes:</b> {pred_naive:.1f} ⭐ (Confidence: {prob_naive:.2%})<br>
                <b>SVC:</b> {pred_svc:.1f} ⭐ (Confidence: {prob_svc:.2%})<br>
                <b>Neural Network:</b> {pred_nn:.1f} ⭐ (Confidence: {prob_nn:.2%})
                </div>
                """,
                unsafe_allow_html=True
            )

            ratings_df = pd.DataFrame({
                'Model': ['Naive Bayes', 'SVC', 'Neural Network'],
                'Rating': [pred_naive, pred_svc, pred_nn]
            })
            fig = px.bar(ratings_df, x='Model', y='Rating', title=f'Predicted Ratings for {label} Review',
                         range_y=[0, 5], color='Model', text='Rating')
            fig.update_traces(texttemplate='%{text:.1f}', textposition='auto')
            st.plotly_chart(fig)

elif page == "Model & Data Insights":
    st.title("📊 Model & Data Insights")
    st.markdown("""
    Explore the performance of our machine learning models and insights from the Amazon Product Reviews Dataset.
    """)

    # Model performance
    st.subheader("Model Performance")
    st.markdown("Compare the accuracy and other metrics of Naive Bayes, SVC, and Neural Network models.")

    # Accuracy comparison bar chart
    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df.round(3)
    st.write("**Performance Metrics**")
    st.dataframe(metrics_df, use_container_width=True)

    fig = px.bar(metrics_df, x=metrics_df.index, y='Accuracy', title='Model Accuracy Comparison',
                 text='Accuracy', color=metrics_df.index)
    fig.update_traces(texttemplate='%{text:.3f}', textposition='auto')
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig)

    # Confusion matrix heatmaps
    st.subheader("Confusion Matrices")
    st.markdown("These heatmaps show how well each model predicts ratings (1 to 5 stars). High values on the diagonal indicate correct predictions.")
    for model_name, cm in confusion_matrices.items():
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=['1.0', '2.0', '3.0', '4.0', '5.0'],
            y=['1.0', '2.0', '3.0', '4.0', '5.0'],
            colorscale='Blues',
            showscale=True
        )
        fig.update_layout(title=f'Confusion Matrix: {model_name}', xaxis_title='Predicted', yaxis_title='True')
        st.plotly_chart(fig)

    # Precision-recall curves (simplified for one class, e.g., 5.0 vs. others)
    st.subheader("Precision-Recall Curves")
    st.markdown("These curves show the trade-off between precision and recall for predicting 5-star reviews.")
    # Note: Requires test data; here we simulate with cached models
    X_test = vectorizer.transform(data['reviews.text'].apply(clean_data))
    y_test = (data['reviews.rating'] == 5.0).astype(int)  # Binary: 5.0 vs. others
    for model_name, model in [('Naive Bayes', model_naive), ('SVC', model_SVC), ('Neural Network', model_NN)]:
        y_scores = model.predict_proba(X_test)[:, -1]  # Probability for 5.0
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        pr_df = pd.DataFrame({'Recall': recall, 'Precision': precision})
        fig = px.line(pr_df, x='Recall', y='Precision', title=f'Precision-Recall Curve: {model_name}',
                      range_x=[0, 1], range_y=[0, 1])
        st.plotly_chart(fig)

    # Data insights
    st.subheader("Dataset Insights")
    st.markdown("Understand the distribution of ratings and key words in the Amazon Product Reviews Dataset.")

    # Rating distribution bar plot
    rating_counts = data['reviews.rating'].value_counts().sort_index()
    rating_df = pd.DataFrame({'Rating': rating_counts.index, 'Count': rating_counts.values})
    fig = px.bar(rating_df, x='Rating', y='Count', title='Rating Distribution in Dataset',
                 text='Count', color='Rating')
    fig.update_traces(texttemplate='%{text}', textposition='auto')
    st.plotly_chart(fig)

    # Word cloud of top TF-IDF features
    st.subheader("Top Words Influencing Predictions")
    feature_names = vectorizer.get_feature_names_out()
    naive_coefs = model_naive.feature_log_prob_.max(axis=0)
    top_features = pd.Series(naive_coefs, index=feature_names).sort_values(ascending=False)[:50]
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_features)
    
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Models trained on Amazon Product Reviews Dataset | Fmachenaud")
