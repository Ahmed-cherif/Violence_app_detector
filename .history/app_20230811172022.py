import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the pre-trained model
best_model = load_model("best_model.h5")

# Load tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(["dummy"])  # We just need to initialize it for consistency

# Streamlit app
def main():
    st.title("Sentiment Analysis App")

    # User input
    user_input = st.text_area("Enter text for sentiment analysis:", "")

    if st.button("Analyze"):
        if user_input:
            # Preprocess user input
            user_input = [user_input]
            user_sequences = tokenizer.texts_to_sequences(user_input)
            user_sequences = pad_sequences(user_sequences, maxlen=best_model.input_shape[1])

            # Predict sentiment
            prediction = best_model.predict(user_sequences)[0][0]

            sentiment = "Positive" if prediction >= 0.5 else "Negative"

            st.write("Sentiment:", sentiment)
            st.write("Confidence:", prediction)

if __name__ == "__main__":
    main()
