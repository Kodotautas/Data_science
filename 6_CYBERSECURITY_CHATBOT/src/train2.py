import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_preprocess import df_to_list, max_words

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

#short conversations for testing (list of lists)
conversations = conversations[:10]

# ------------------------- PREPARE DATA FOR TRAINING ------------------------ #
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Tokenize the conversations
input_ids = []
attention_mask = []
for conversation in conversations:
    input_id = []
    attention_mask_ = []
    for sentence in conversation:
        encoded = tokenizer.encode(sentence, add_special_tokens=True, max_length=max_lenght, pad_to_max_length=True)
        input_id.append(encoded)
        attention_mask_.append([1] * len(encoded))
    input_ids.append(input_id)
    attention_mask.append(attention_mask_)

# Convert the lists to tensors
input_ids = torch.tensor(input_ids)
attention_mask = torch.tensor(attention_mask)

#check input_ids and attention_mask
print(input_ids.shape)
print(attention_mask.shape)

# decode the input_ids to check if the encoding is correct
print(tokenizer.decode(input_ids[1][0], skip_special_tokens=True))
print(tokenizer.decode(input_ids[1][1], skip_special_tokens=True))


# ------------------------------- TRAIN THE MODEL ---------------------------- #
# Train the model
epochs = 15
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
losses = []
print('Training...')
for i in range(epochs):
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs[0]
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    print('Loss: {}'.format(loss.item()))


# ------------------------------- SAVE THE MODEL ----------------------------- #
# Save the model
model.save_pretrained(path + '/models/dialogpt')

# Save the tokenizer
tokenizer.save_pretrained(path + '/models/dialogpt')
print('Model and tokenizer saved')


# ------------------------------- TEST MODEL OUTPUT -------------------------- #
# Test the model and calculte accuracy for each conversation
model.eval()
with torch.no_grad():
    for i, conversation in enumerate(conversations):
        print('Conversation: {}'.format(i))
        for j, sentence in enumerate(conversation):
            if j == 0:
                input_ids = tokenizer.encode(sentence, return_tensors='pt')
                input_ids = input_ids.to(device)
                generated = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
                generated = generated[0].tolist()
                generated = tokenizer.decode(generated, skip_special_tokens=True)
                print('User: {}'.format(sentence))
                print('Bot: {}'.format(generated))
                print('Actual: {}'.format(conversation[j+1]))

# calculate nlp accuracy
def calculate_accuracy(model, tokenizer, conversations):
    model.eval()
    with torch.no_grad():
        for i, conversation in enumerate(conversations):
            print('Conversation: {}'.format(i))
            for j, sentence in enumerate(conversation):
                if j == 0:
                    input_ids = tokenizer.encode(sentence, return_tensors='pt')
                    input_ids = input_ids.to(device)
                    generated = model.generate(input_ids, max_length=max_lenght, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
                    generated = generated[0].tolist()
                    generated = tokenizer.decode(generated, skip_special_tokens=True)
        
print('Accuracy: {}'.format(calculate_accuracy(model, tokenizer, conversations)))

