import os
import pandas as pd
from src.generate_response import chatbot_response
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import warnings
warnings.filterwarnings("ignore")

path = os.getcwd()


# -------------------------------- LOAD MODEL -------------------------------- #
# Load the trained model from disk
model = AutoModelForCausalLM.from_pretrained(path + '/models/dialogpt')
# Initialize the tokenizer and the model dialogpt-large
tokenizer = AutoTokenizer.from_pretrained(path + '/models/dialogpt')


# Check if a GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Move the model to the device
model = model.to(device)


# ---------------------------------- CHATBOT --------------------------------- #
# create a loop to test the chatbot until the user types "bye"
while True:
    # Get the user input
    user_input = input('You: ')

    # Check if the user input is a question
    if user_input[-1] != '?':
        print('Chatbot: Please ask a question.')
        continue

    # Check if the user input is "bye"
    if user_input.lower() == 'bye':
        print('Chatbot: Bye')
        break

    # Generate a response
    response = chatbot_response(user_input, model, tokenizer, device)

    # Print the response
    print('Chatbot: ' + response)