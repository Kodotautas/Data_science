import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_preprocess import df_to_list, max_words
# add padding to tensors
from torch.nn import functional as F

import warnings
warnings.filterwarnings("ignore")

path = os.getcwd()
# one level up
path = os.path.dirname(path)


# ------------------------------ PREPROCESS DATA ----------------------------- #
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-large')

# Set the device to run on (CPU or GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)

df = pd.read_excel(path + '/data/security_faq.xlsx')
max_lenght = max_words(df)
conversations = df_to_list(df)

#short conversations for testing
conversations = conversations[:5]

# ------------------------- PREPARE DATA FOR TRAINING ------------------------ #
# Tokenize the conversations
input_ids = []
attention_mask = []
for conversation in conversations:
    input_id = tokenizer.encode(conversation, add_special_tokens=True, max_length=max_lenght, pad_to_max_length=True)
    attention_m = [1] * len(input_id)
    input_ids.append(torch.tensor(input_id))
    attention_mask.append(torch.tensor(attention_m))

# Convert the lists to tensors
input_ids = torch.stack(input_ids)
attention_mask = torch.stack(attention_mask)

# Move the tensors to the device
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

# check decoded input and attention mask
print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
print(attention_mask[0])


# ------------------------------- TRAIN THE MODEL ---------------------------- #
# Train the model
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
losses = []
print('Training...')
for epoch in range(1):
    total_loss = 0
    for i in range(len(input_ids)):
        optimizer.zero_grad()
        outputs = model(input_ids[i].unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0), labels=input_ids[i].unsqueeze(0))
        loss, logits = outputs[:2]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss / len(input_ids))
    print('Epoch: {}, Loss: {}'.format(epoch, total_loss / len(input_ids)))


# ------------------------------- SAVE THE MODEL ----------------------------- #
# Save the model
model.save_pretrained(path + '/models/dialogpt')

# Save the tokenizer
tokenizer.save_pretrained(path + '/models/dialogpt')
print('Model and tokenizer saved')


# ------------------------------- TEST AND EVALUATE -------------------------- #
# Test the model
model.eval()
print('Testing...')
with torch.no_grad():
    for i in range(len(input_ids)):
        question = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        answer = model.generate(input_ids[i].unsqueeze(0), max_length=100, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
        print('Question: {}'.format(question))
        print('Answer: {}'.format(tokenizer.decode(answer[0], skip_special_tokens=True)))
        print('')


# Calculate the accuracy
def accuracy():
    correct = 0
    for i in range(len(input_ids)):
        answer = model.generate(input_ids[i].unsqueeze(0), max_length=max_lenght, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
        if tokenizer.decode(answer[0], skip_special_tokens=True) == tokenizer.decode(input_ids[i], skip_special_tokens=True):
            correct += 1
    return correct / len(input_ids)

print('Accuracy: {}'.format(accuracy()))