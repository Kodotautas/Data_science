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
conversations = conversations[2:4]
print(conversations)


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
        encoded = tokenizer.encode(sentence, add_special_tokens=True, pad_to_max_length=True, truncation=True, max_length=max_lenght)
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
print('Q:', tokenizer.decode(input_ids[1][0], skip_special_tokens=True))
print('A:', tokenizer.decode(input_ids[1][1], skip_special_tokens=True))


# ------------------------------- TRAIN THE MODEL ---------------------------- #
# Train the model
epochs = 5
batch_size = 1
learning_rate = 1e-5

# Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model
for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    for i in range(0, len(input_ids), batch_size):
        # Get the inputs and labels
        input_ids_ = input_ids[i:i+batch_size]
        attention_mask_ = attention_mask[i:i+batch_size]
        labels = input_ids_.clone()
        labels[:, :-1] = input_ids_[:, 1:]

        # Move the tensors to the device
        input_ids_ = input_ids_.to(device)
        attention_mask_ = attention_mask_.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(input_ids_, attention_mask=attention_mask_, labels=labels)
        loss = outputs[0]

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print the loss
        print('Loss: {}'.format(loss.item()))


# ------------------------------- SAVE THE MODEL ----------------------------- #
# Save the model
model.save_pretrained(path + '/models/dialogpt')

# Save the tokenizer
tokenizer.save_pretrained(path + '/models/dialogpt')
print('Model and tokenizer saved to disk.')


# ------------------------------- TEST MODEL OUTPUT -------------------------- #
# Test the model and calculte accuracy for each conversation
model.eval()
with torch.no_grad():
    for i, conversation in enumerate(conversations):
        print('Conversation: {}'.format(i+1))
        for j, sentence in enumerate(conversation):
            if j == 0:
                input_ids = tokenizer.encode(sentence, return_tensors='pt')
                attention_mask = torch.ones_like(input_ids)  # create attention mask
                input_ids = input_ids.to(device)
                generated = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
                generated = generated[0].tolist()
                generated = tokenizer.decode(generated, skip_special_tokens=True)
                print('User: {}'.format(sentence))
                print('Bot: {}'.format(generated))
                # print('Actual: {}'.format(conversation[j+1]))


# calculate accuracy
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