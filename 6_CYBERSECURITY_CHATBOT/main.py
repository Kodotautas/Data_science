import os
import random
import pandas as pd
from src.data_preporcess import df_to_tuple, preprocess_text, create_vocabulary
from models.model import ChatBot
from src.train import train
import torch
import torch.nn as nn
import torch.optim as optim

import warnings
warnings.filterwarnings("ignore")

cwd = os.getcwd()


# ------------------------------ DATA PREPROCESSING ------------------------------ #
# read the cybersecurity faq excel file
df = pd.read_excel(cwd + '/data/security_faq.xlsx')

# convert df to tuple of questions and answers with '\n' as delimiter
faq = df_to_tuple(df)

# preprocess the text
train_data = preprocess_text(faq)

# check vocabulary size
vocabulary = create_vocabulary(train_data)


# ------------------------------ MODEL TRAINING ------------------------------ #
# Set hyperparameters
vocab_size = 10000
embedding_dim = 200
hidden_dim = 200
learning_rate = 0.001
batch_size = 128
num_epochs = 10

# Initialize the model and the optimizer
model = ChatBot(vocab_size, embedding_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train(model, optimizer, train_data, vocab_size, embedding_dim, hidden_dim, learning_rate, batch_size, num_epochs)