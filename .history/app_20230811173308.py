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
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Pad sequences
max_length = max([len(seq) for seq in train_sequences + test_sequences])
train_sequences = pad_sequences(train_sequences, maxlen=max_length)
test_sequences = pad_sequences(test_sequences, maxlen=max_length)

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
            user_sequences = pad_sequences(user_sequences, maxlen=max_length)

            # Predict sentiment
            prediction = best_model.predict(user_sequences)[0][0]

            sentiment = "Positive" if prediction >= 0.5 else "Negative"

            st.write("Sentiment:", sentiment)
            st.write("Confidence:", prediction)

if __name__ == "__main__":
    main()
