# Sentiment Analysis Model Evaluation Notebook

This Jupyter Notebook contains code to evaluate sentiment analysis models' performance, visualize results, and determine the best model based on prediction scores and reality values.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)
- [License](#license)

## Introduction

This notebook demonstrates how to load sentiment analysis models, preprocess text data, make predictions, and evaluate the models' performance. It includes visualizations to help analyze the relationships between prediction scores, sentiments, reality values, and model characteristics.

## Prerequisites

Before using this notebook, make sure you have the following:

1. Python 3.x installed.
2. Jupyter Notebook installed.
3. Necessary libraries installed:
   - pandas
   - matplotlib
   - seaborn
   - tensorflow (for model loading and prediction)

You can install the required libraries using the following command:



## Usage

1. Place your sentiment analysis model files (e.g., `sentiment_model0.h5`, `sentiment_model1.h5`, etc.) in the same directory as this notebook.
2. Update the `new_text_examples` list with your desired text examples.
3. Run each cell in the notebook to perform model evaluation, analysis, and visualization.
4. View the generated plots and read the printed results to understand model performance.

## Results

The notebook generates various visualizations and provides results to help you analyze the performance of different sentiment analysis models. The following are some key insights:

- Stacked bar plots illustrate the distribution of predicted sentiments by model.
- Box plots showcase the distribution of prediction scores for different predicted sentiments.
- Pairplot displays relationships between prediction score, sentiment, model number, and reality.
- Violin plots show the distribution of prediction scores by predicted sentiment and model name.

The notebook identifies the best model for each sentiment based on Mean Absolute Differences (MAD) between prediction scores and reality. It also determines the overall best model.

## Dependencies

The following libraries are used in this notebook:

- pandas
- matplotlib
- seaborn
- tensorflow (for loading and using sentiment analysis models)

## License

This notebook is provided under the [MIT License](LICENSE). You are free to use and modify this notebook for your own purposes.

# Streamlit Sentiment Analysis App

Welcome to the Streamlit Sentiment Analysis App! This app allows you to perform sentiment analysis on text input, predicting whether the input text contains violent content or not. It utilizes a pre-trained deep learning model for accurate predictions.

## Table of Contents

- [Prerequisites](#prerequisites)
- [App Functionality](#app-functionality)
- [How to Use](#how-to-use)
- [Dependencies](#dependencies)
- [License](#license)

## Prerequisites

Before using the Streamlit app, make sure you have the following dependencies installed:

- `streamlit`: Streamlit is a Python library used to create interactive web applications with minimal code.

You can install Streamlit using the following command:
```bash
pip install streamlit

App Functionality
The Streamlit Sentiment Analysis App offers the following features:

Loading Pre-trained Model: The app loads a pre-trained deep learning model for sentiment analysis. This model has been trained to classify input text as containing violence or not.

Reading Data: The app reads data from a source, such as an Excel file (Dataset.xlsx), which is similar to the data used for training the sentiment analysis model.

Data Preprocessing: The input data from the source is preprocessed using the Tokenizer and pad_sequences functions. This prepares the input text for accurate model predictions.

User Interface: The app provides a user-friendly interface with the following components:

Title: Displays "Détection de Violence" to convey the app's purpose.
Text Area: Allows users to enter text for sentiment analysis.
"Prédire" Button: Initiates the sentiment analysis prediction process.
Prediction: Upon clicking the "Prédire" button, the input text is preprocessed, and the model predicts whether violence is detected in the text. The prediction is presented to the user.

How to Use
Ensure you have the necessary dependencies installed.

Place the pre-trained model file (best_model2.h5) and the data source file (Dataset.xlsx) in the same directory as the Streamlit app script.

Run the Streamlit app by executing the script in your terminal:
streamlit run your_app_script.py
Replace your_app_script.py with the actual name of your Streamlit script.

Access the Streamlit app through your web browser. Enter text in the provided text area and click the "Prédire" button to receive a sentiment analysis prediction.

Dependencies
The Streamlit Sentiment Analysis App relies on the following libraries:

streamlit
numpy
tensorflow (for loading the pre-trained model)
pandas
License
This Streamlit Sentiment Analysis App is provided under the MIT License. You are free to use and modify this app for your own purposes.
