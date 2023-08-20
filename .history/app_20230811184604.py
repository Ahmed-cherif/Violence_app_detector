import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Load the best model
best_model = load_model("sentiment_model1.h5")

# Load the saved tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Title and description with Bootstrap styling
st.title("Medical Violence Detection")
st.markdown(
    """<style>
        .title {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 2rem;
        }
        .description {
            text-align: center;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
    </style>""",
    unsafe_allow_html=True
)
st.markdown('<p class="title">Medical Violence Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="description">Enter a medical text to determine if it contains medical violence.</p>', unsafe_allow_html=True)

# Text input
text_input = st.text_area("Enter medical text:", "")

# Preprocess the input text
max_length = best_model.input_shape[1]  # Use the same max_length as in your model
sequences = tokenizer.texts_to_sequences([text_input])
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Make prediction
if st.button("Predict"):
    prediction = best_model.predict(padded_sequences)
    result = "Violence médicale" if prediction > 0.5 else "Non violence médicale"
    st.markdown(
        f"""<div class="result">
                <p>Prediction: <strong>{result}</strong></p>
            </div>""",
        unsafe_allow_html=True
    )
    st.write(prediction)
