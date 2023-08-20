import streamlit as st
import numpy as np  
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Page configuration
st.set_page_config(page_title="Medical Violence Detection", page_icon="🩺", layout="wide")

# Style
style = """
<style>
.title {
  text-align: center;
  font-size: 2.5rem;
  margin-bottom: 1.5rem;  
}

.description {
  text-align: center;
  font-size: 1.2rem; 
  margin-bottom: 2rem;
}

.result {
  text-align: center;
  font-size: 1.5rem;
} 
</style>
"""

st.markdown(style, unsafe_allow_html=True)

# Page title and description
st.markdown('<p class="title">Medical Violence Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="description">Enter a medical text to determine if it contains medical violence.</p>', unsafe_allow_html=True)

# Load model and tokenizer
best_model = load_model("sentiment_model1.h5")
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
# Get user input   
text_input = st.text_area("Enter medical text:", "") 

# Preprocess input
max_length = best_model.input_shape[1]
sequences = tokenizer.texts_to_sequences([text_input]) 
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Make prediction
if st.button("Predict"):
    prediction = best_model.predict(padded_sequences)
    
    if prediction > 0.5:
        result = "Violence médicale" 
        st.markdown(f'<p style="color: red; font-size: 20px; text-align:center;">{result}</p>', unsafe_allow_html=True)
    else:
        result = "Non violence médicale"
        st.markdown(f'<p style="color: green; font-size: 20px; text-align:center;">{result}</p>', unsafe_allow_html=True)
        
    st.write(f"Prediction: {np.round(prediction[0][0], 2)}")