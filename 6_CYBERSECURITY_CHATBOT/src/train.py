import os
from transformers import BertTokenizer
from data_preprocess import df_to_tuple, convert_to_jsonl
import pandas as pd
import torch

cwd = os.getcwd()

# Initialize the tokenizer and the model dialogpt-large
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# read the cybersecurity faq excel file
# df = pd.read_excel(cwd + '/data/security_faq.xlsx')

# # convert the dataframe to a list of tuples
# data = df_to_tuple(df)
# data = convert_to_jsonl(data)

data = [
    {"input": "Why do I need to worry about Cyber Security?", "output": "Cyber Security protects unauthorized access and or criminal use of your data."},
    {"input": "Should I have a different password for every website?", "output": "Yes. If you use the same password on every web- site and someone gets access to it."},
    {"input": "How can I remember all of my passwords?", "output": "You can use a password manager. There are free options and low cost ones."},
]

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

from transformers import BertForMaskedLM

# Initialize the model and optimizer
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
optimizer = torch.optim.Adam(model.parameters())

import torch.nn as nn

# Define the loss function as the cross-entropy loss
loss_fn = nn.CrossEntropyLoss()

# Define the evaluation metric as the accuracy
def accuracy(predictions, targets):
  predictions = predictions.argmax(dim=-1)
  return (predictions == targets).float().mean()

# Set the number of epochs and the batch size
num_epochs = 10
batch_size = 32

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
    predictions = model(input_sequences, labels=output_sequences)[1]
    loss = loss_fn(predictions.view(-1, predictions.size(-1)), output_sequences.view(-1))
    acc = accuracy(predictions, output_sequences)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Update the running loss and accuracy
    running_loss += loss.item()
    running_acc += acc.item()

    # Print the loss and accuracy for the epoch
    print(f'Epoch: {epoch+1}, Loss: {running_loss/len(input_sequences):.4f}, Accuracy: {running_acc/len(input_sequences):.4f}')

# Save the model
model.save_pretrained('model')