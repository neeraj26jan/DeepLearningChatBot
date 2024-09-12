import random
import pandas as pd 
import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import string
import tensorflow as tf 
import os
import csv

# Suppress TensorFlow logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Load the trained model and necessary preprocessing tools
model = load_model('chatbot_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pickle', 'rb') as handle:
    le = pickle.load(handle)

# Define Parameters
max_len = 6 # Ensure this matches the max length used during training
data_file_path = 'chat_data.csv'
log_file_path = 'chat_logs.csv'

# Clean the user input text
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'\d+','', text)
    text = text.translate(str.maketrans('','', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text

# Generate a response from the chatbot model based on user input
def get_response(user_input):
    cleaned_input = clean_text(user_input)
    seq = tokenizer.texts_to_sequences([cleaned_input])
    padded_seq = pad_sequences(seq, maxlen=max_len)

    # Predict response probabilities
    pred = model.predict(padded_seq, verbose=0)[0]

    # Get the indices of the top 3 predictions
    top_indices = np.argsort(pred)[-3:]
    top_probs = pred[top_indices]

    # Choose the response with the highest probability or a random top response
    if (top_probs[-1] - top_probs[-2]) > 0.00019: # Threshold for confidence
        response_index = top_indices[-1]
    else:
        response_index = np.random.choice(top_indices)

    response = le.inverse_transform([response_index])[0]
    return response

# Save the user input and bot response to a CSV log file
def save_chat_log(user_input, response, log_file_path='chat_logs.csv'):

    # Create the file if it doesn't exist
    if not os.path.exists(log_file_path):
        with open(log_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(['user_input', 'bot_response'])

    # Append the new chat log entry
    try:
        with open(log_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow([user_input, response])
    except Exception as e:
        print(f"An error occurred while writing to the log file: {e}")

    # Validate the CSV file structure
    try:
        pd.read_csv(log_file_path)   
    except pd.errors.ParseError as e:
        print(f"CSV file parsing error detected: {e}")
        # Handle the error if needed

# Run and interactive chatbot session
def interactive_chatbot():
    print("Chatbot is running. Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        else:
            response = get_response(user_input)
            print("Chatbot:", response)
            save_chat_log(user_input, response)

# Run the interactive chatbot
interactive_chatbot()
