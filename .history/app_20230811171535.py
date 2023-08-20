import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# Load the pre-trained model
model = load_model("best_model2.h5")

# Read the data from the same source you used for training (e.g., Excel file)
df = pd.read_excel("Dataset.xlsx")

# Preprocess the data
df['data'] = df['data'].astype(str)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['data'])
sequences = tokenizer.texts_to_sequences(df['data'])
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length)

# Streamlit app
st.title('Détection de Violence')

text = st.text_area('Entrez du texte:')
if st.button('Prédire'):
    # Preprocess the input text
    preprocessed_text = tokenizer.texts_to_sequences([text])
    preprocessed_text = pad_sequences(preprocessed_text, maxlen=max_length)
    # Make prediction
    prediction = model.predict(preprocessed_text)
    print(prediction[0][0])
    prediction_label = "Violence détectée" if prediction[0][0] >= 0.5 else "Pas de violence détectée"
    st.write(prediction_label)