import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the best model
best_model = load_model("best_model.h5")

# Load the saved Tokenizer
tokenizer = joblib.load("tokenizer.joblib")

# Streamlit app title and description
st.title("Medical Sentiment Analysis")
st.write("Enter medical text to analyze its sentiment.")

# Text input for user to enter new examples
user_input = st.text_area("Enter medical text:", value="", height=150)

# Submit button
submit_button = st.button("Analyze Sentiment")

if submit_button and user_input:
    # Tokenization and padding
    sequences = tokenizer.texts_to_sequences([user_input])
    max_length = best_model.input_shape[1]
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    # Make predictions
    predicted_label = best_model.predict(padded_sequences)[0][0]

    # Determine sentiment based on prediction
    sentiment = "Violence médicale" if predicted_label > 0.5 else "Non violence médicale"

    # Display prediction and sentiment with color-coding
    result_color = "red" if predicted_label > 0.5 else "green"
    st.markdown("---")
    st.subheader("Sentiment Analysis Result")
    st.write(f"Predicted Sentiment: <span style='color:{result_color}'>{sentiment}</span>", unsafe_allow_html=True)
    st.write(f"Prediction Probability: {predicted_label:.4f}")
    st.markdown("---")
