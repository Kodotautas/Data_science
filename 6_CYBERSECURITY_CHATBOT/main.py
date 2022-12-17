import os
import pandas as pd
from src.data_preprocess import df_to_tuple, convert_to_jsonl
from src.generate_response import chatbot_response
import torch
import torch.nn as nn
from transformers import BertForMaskedLM
from transformers import BertTokenizer

import warnings
warnings.filterwarnings("ignore")

cwd = os.getcwd()


# ------------------------------ DATA PREPROCESSING ------------------------------ #
# read the cybersecurity faq excel file
df = pd.read_excel(cwd + '/data/security_faq.xlsx')

# convert the dataframe to a list of tuples
data = df_to_tuple(df)
convert_to_jsonl(data)

# Initialize the tokenizer and the model dialogpt-large
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# -------------------------------- LOAD MODEL -------------------------------- #
# Load the trained model
model = BertForMaskedLM.from_pretrained('./models/trained_model')

# Check if a GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Move the model to the device
model = model.to(device)


# ------------------------------- TEST CHATBOT ------------------------------- #
# create a loop to test the chatbot until the user types "bye"
while True:
    prompt = input("Enter your message: ")
    if prompt == "bye" or prompt == "Bye":
        break
    response = chatbot_response(prompt, tokenizer, model, device)
    print(f"Chatbot: {response}")
