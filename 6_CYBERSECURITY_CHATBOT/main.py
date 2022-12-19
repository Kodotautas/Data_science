import os
from src.generate_response import chatbot_response
import pandas as pd

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
# Start the chatbot
model.eval()
print('Predicting...')
with torch.no_grad():
    while True:
        question = input('Enter your question: ')
        if question == 'quit':
            break
        question = tokenizer.encode(question, return_tensors='pt').to(device)
        answer = model.generate(question, max_length=100, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
        print('Answer: {}'.format(tokenizer.decode(answer[0], skip_special_tokens=True)))
        print('')