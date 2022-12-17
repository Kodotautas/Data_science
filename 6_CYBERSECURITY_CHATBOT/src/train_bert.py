import json
import os
from transformers import BertTokenizer
from transformers import BertForMaskedLM
from transformers import logging
from src.data_preprocess import df_to_tuple, convert_to_jsonl, accuracy
import pandas as pd
import torch
import torch.nn as nn

logging.set_verbosity_warning()

cwd = os.getcwd()


# ----------------------------- TRANSFORM DATASET ---------------------------- #
# Initialize the tokenizer and the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# read the cybersecurity faq excel file
df = pd.read_excel(cwd + '/data/security_faq.xlsx')

# convert to json lines
data = df_to_tuple(df)
convert_to_jsonl(data, cwd)

# Load the data from the JSONL file
with open(cwd + '/data/security_faq.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]


# ---------------------- TOKENIZE AND CONVERT TO TENSORS --------------------- #
# Tokenize the input and output
input_sequences = [tokenizer.encode(datum['input'], add_special_tokens=True) for datum in data]
output_sequences = [tokenizer.encode(datum['output'], add_special_tokens=True) for datum in data]

# Pad the sequences to the same length
max_length = max(len(sequence) for sequence in input_sequences + output_sequences)
input_sequences = [sequence + [0] * (max_length - len(sequence)) for sequence in input_sequences]
output_sequences = [sequence + [0] * (max_length - len(sequence)) for sequence in output_sequences]

# Convert the sequences to tensors
input_tensors = torch.tensor(input_sequences)
output_tensors = torch.tensor(output_sequences)


# ----------------------------------- MODEL ---------------------------------- #
# Initialize the model and optimizer
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
optimizer = torch.optim.Adam(model.parameters())

# Define the loss function as the cross-entropy loss
loss_fn = nn.CrossEntropyLoss()

# Check if a GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Move the input and output tensors and the model to the device
input_tensors = input_tensors.to(device)
output_tensors = output_tensors.to(device)
model = model.to(device)

# Modify the loss function to be computed on the device
loss_fn = loss_fn.to(device)

# Set the number of epochs and the batch size
num_epochs = 10
batch_size = 32


# ------------------------------- TRAINING LOOP ------------------------------ #
print("Training the model...")
# Iterate over the epochs
for epoch in range(num_epochs):
  # Initialize the running loss and accuracy
  running_loss = 0.0
  running_acc = 0.0

  # Iterate over the batches of data
  for i in range(0, len(input_tensors), batch_size):
    # Extract the input and output sequences
    input_sequences = input_tensors[i:i+batch_size]
    output_sequences = output_tensors[i:i+batch_size]

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    predictions = model(input_sequences.to(device), labels=output_sequences.to(device))[1]
    loss = loss_fn(predictions.view(-1, predictions.size(-1)).to(device), output_sequences.view(-1).to(device))
    acc = accuracy(predictions.to(device), output_sequences.to(device))

    # Backward pass
    loss.backward()
    optimizer.step()

    # Update the running loss and accuracy
    running_loss += loss.item()
    running_acc += acc.item()
    
  # Print the loss and accuracy for the epoch
  print(f"Epoch {epoch+1} - Loss: {running_loss/len(input_tensors):.4f} - Accuracy: {running_acc/len(input_tensors):.4f}")


# ------------------------------- SAVE THE MODEL ----------------------------- #
# Save the model
model.save_pretrained(cwd + '/models/bert')