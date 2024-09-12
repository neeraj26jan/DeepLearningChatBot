import numpy as np 
import pandas as pd 
import re
import string
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.utils import to_categorical
import pickle
import tensorflow as tf 
import csv
import warnings
import os

#Suppress Tensorflow warnings and logs
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

#Check if chat_logs.csv exists
if os.path.exists('chat_logs.csv'):
    # Load and transform the chat logs
    chat_logs_df = pd.read_csv('chat_logs.csv')
    transformed_data = []

    # Transform chat logs by shifting bot responses to become user inputs and next user inputs to become bot responses
    for i in range(len(chat_logs_df) - 1):
        user_input = chat_logs_df.iloc[i]['bot_response']
        bot_response = chat_logs_df.iloc[i + 1]['user_input']
        transformed_data.append({'userinput': user_input, 'bot_response': bot_response})

        # Conver the transformed data into a DataFrame
        transformed_df = pd.DataFrame(transformed_data)

        # Load the original chat data
        chat_data_df = pd.read_csv('chat_data.csv')

        # Combine original chat data with transfored chat data
        combined_df = pd.concat([chat_data_df, transformed_df])

        # Remove any duplicate rows
        combined_df.drop_duplicates(inplance=True)

        # Save the updated chat data to a CSV file with all fields quoted
        combined_df.to_csv('chat_data.csv', index=False, quotechar='"', quoting=csv.QUOTE_ALL)
else:
    # If chat_logs.csv does not exist, use the existing chat_data.csv directly
    print("chat_logs.csv not found. Using existing chat_dat.csv")

    # Load the updated ddataset for training
    data = pd.read_csv('chat_data.csv') 

    # Seperate user inputs and bot responses
    texts = data['user_input'].values
    responses = data['bot_response'].values

    # Tokenize the user inputs
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    x = tokenizer.texts_to_sequences(texts) 

    # Pad the sequences to ensure uniform input size
    max_len = max(len(x) for x in x)
    x = pad_sequences(x, maxlen=max_len)

    # Encode the bot responses
    le = LabelEncoder()
    y = le.fit_transform(responses)
    y = to_categorical(y)

    # Define the neural network model
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_len),
        LSTM(128),
        Dense(len(le.classes_), activation='softmax')
    ])    

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x, y, epochs=6, verbose=0)

    # Save the trained model, tokenizer and label encoder
    model.save('chatbot_model.h5')
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('label_encoder.pickle', 'wb') as handle:
        pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Training complete. Model and tokenizer saved.")    