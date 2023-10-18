# Libraries
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import numpy as np

# Suppress some warnings
tf.get_logger().setLevel('ERROR')

# Load the trained model
model_path = r"H:\DS Projects\NLP Sentiment anlysis\data\models\sentiment_analysis_distilbert_model"  # Adjust with your model path
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to tokenize sentences
def tokenize_sentences(tokenizer, sentences, max_length=128):
    return tokenizer(sentences, max_length=max_length, padding=True, truncation=True, return_attention_mask=True, return_tensors='tf')

# Prediction function for custom text input
def predict_sentiment_detailed(model, tokenizer, sentence):
    # Tokenize the sentence
    inputs = tokenize_sentences(tokenizer, [sentence])

    # Make a prediction
    outputs = model.predict({'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']})

    # The output consists of logits, we need to convert these to probabilities using softmax
    probabilities = tf.nn.softmax(outputs.logits[0]).numpy()  # Using softmax to get probabilities

    # Assuming the classes are 'negative', 'neutral', and 'positive'
    class_names = ['negative', 'neutral', 'positive']  # If your model was trained differently, the class order might differ

    # Create a dictionary to hold the class names and their probabilities
    sentiment_scores = {class_name: prob for class_name, prob in zip(class_names, probabilities)}

    # You can also identify the highest scoring class as the predicted sentiment
    predicted_class = class_names[np.argmax(probabilities)]  # Getting the class with the highest probability

    return sentiment_scores, predicted_class

# Example of predicting sentiment of custom text input with detailed scores
custom_text = "I bought this mixture yesterday, beleive me it is working on it's all modes. Well I must say that, this product is Awesome"
sentiment_scores, predicted_class = predict_sentiment_detailed(model, tokenizer, custom_text)

print(f'Sentiment scores: {sentiment_scores}')
print(f'Predicted sentiment class: {predicted_class}')
