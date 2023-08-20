import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the best model
best_model = load_model("sentiment_model1.h5")

# Title and description
st.title("Medical Violence Detection")
st.write("Enter a medical text to determine if it contains medical violence.")

# Text input
text_input = st.text_area("Enter medical text:", "")

# Preprocess the input text
max_length = 100  # You can adjust this based on your model's input length
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_input])
input_sequence = tokenizer.texts_to_sequences([text_input])
input_sequence = pad_sequences(input_sequence, maxlen=max_length)

# Make prediction
if st.button("Predict"):
    prediction = best_model.predict(input_sequence)
    if prediction > 0.5:
        st.write("Prediction: Violence médicale")
    else:
        st.write("Prediction: Non violence médicale")
