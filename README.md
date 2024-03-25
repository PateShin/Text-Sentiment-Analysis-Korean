# Text-Sentiment-Analysis-Korean# Sentiment Analysis on Naver Movie Reviews

This project demonstrates sentiment analysis on Naver movie reviews using deep learning models in Python. We utilize TensorFlow and Keras for model building and training, and PyKomoran for Korean text preprocessing.

## Dataset

The dataset comprises reviews from Naver, divided into training and testing sets. Each review is labeled as positive or negative. The datasets are downloaded using URLs pointing to raw text files hosted on GitHub.

**Link to the source github:** https://github.com/e9t/nsmc

- Training set URL: https://raw.github.com/e9t/nsmc/master/ratings_train.txt
- Testing set URL: https://raw.github.com/e9t/nsmc/master/ratings_test.txt

## Preprocessing

Perform the following preprocessing steps:
1. Reading and loading the dataset using Pandas.
2. Dropping rows with missing values.
3. Tokenization and removal of stopwords using PyKomoran.

## Model

Used a LSTM (Long Short-Term Memory) model for sentiment analysis:
- The model is composed of an Embedding layer, a LSTM layer, and a Dense output layer.
- It is compiled with the Adam optimizer and binary cross-entropy loss function.

## Training

- The model is trained on the preprocessed training data for 5 epochs.
- Batch size: 32

## Evaluation

- The trained model is evaluated on the testing set to measure its accuracy and loss.

## Usage

To predict the sentiment of a new review, run the `predict_sentiment` function with the review text as input. It preprocesses the text, feeds it to the trained model, and returns the sentiment prediction.

## Requirements

- TensorFlow
- Keras
- Pandas
- NumPy
- PyKomoran

## Installation

Install the required packages using pip:

```
pip install tensorflow keras pandas numpy PyKomoran
```

# How to Run
Ensure all dependencies are installed. <br>
Download or clone this repository. <br>
Run the Jupyter Notebook or Google Colab. <br>
