import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = load_model("best_model2.h5")

# Streamlit app title and description
st.title("Medical Sentiment Analysis")
st.write("Enter medical text to analyze its sentiment.")

# Text input for user to enter new examples
user_input = st.text_area("Enter medical text:")

# Submit button
submit_button = st.button("Submit")

if submit_button:
    # Tokenization and padding
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([user_input])
    sequences = tokenizer.texts_to_sequences([user_input])
    max_length = model.input_shape[1]
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    # Make predictions
    predicted_label = model.predict(padded_sequences)[0][0]

    # Determine sentiment based on prediction
    sentiment = "Violence médicale" if predicted_label > 0.5 else "Non violence médicale"

    # Display prediction and sentiment
    st.write(f"Predicted Sentiment: {sentiment}")
    st.write(f"Prediction Probability: {predicted_label:.4f}")
