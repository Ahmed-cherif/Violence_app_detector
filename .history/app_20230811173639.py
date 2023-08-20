import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the pre-trained model
best_model = load_model("best_model.h5")

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

# Assuming 'data' is the text column
train_texts = train_data['data'].tolist()
test_texts = test_data['data'].tolist()

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
max_length = max([len(seq) for seq in train_texts + test_texts])  # Use train_texts and test_texts

# Function to preprocess text
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    return padded_sequences

# Streamlit app
def main():
    st.title("Sentiment Analysis App")

    # User input
    user_input = st.text_area("Enter text for sentiment analysis:", "")

    if st.button("Analyze"):
        if user_input:
            # Preprocess user input
            user_sequences = preprocess_text(user_input)

            # Predict sentiment
            prediction = best_model.predict(user_sequences)[0][0]

            sentiment = "Positive" if prediction >= 0.5 else "Negative"

            st.write("Sentiment:", sentiment)
            st.write("Confidence:", prediction)

if __name__ == "__main__":
    main()
