import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from matplotlib import pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf

# Load the dataset from Excel
df = pd.read_excel("Dataset.xlsx")
 
# Shuffle the data
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the shuffled data into training and testing sets
train_size = int(0.8 * len(shuffled_df))
train_data = shuffled_df[:train_size].copy()
test_data = shuffled_df[train_size:].copy()

# Map and convert the labels to integers
label_mapping = {'non violence médicale': 0, 'violence médicale': 1}
train_labels = train_data['resultat'].map(label_mapping).astype(int)
test_labels = test_data['resultat'].map(label_mapping).astype(int)

# Create a Universal Sentence Encoder model
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Preprocess the text data and create embeddings
train_texts = train_data['data'].tolist()
test_texts = test_data['data'].tolist()
train_embeddings = use_model(train_texts)
test_embeddings = use_model(test_texts)

# Load the SavedModel
model = tf.keras.models.load_model('use_model_savedmodel')

# Streamlit UI
st.title("Text Classification with Universal Sentence Encoder")

# Input text box
input_text = st.text_area("Enter text:", "")

if st.button("Predict"):
    try:
        processed_input = use_model([input_text])
        prediction = model.predict(processed_input)[0][0]

        label = "violence médicale" if prediction > 0.5 else "non violence médicale"
        
        st.subheader("Prediction:")
        st.write("Input Text:", input_text)
        st.write("Predicted Label:", label)
        
    except Exception as e:
        st.error("An error occurred while processing the text.")
