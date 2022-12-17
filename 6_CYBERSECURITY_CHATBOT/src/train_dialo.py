import json
import os
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel, AdamW, get_linear_schedule_with_warmup
from src.data_preprocess import df_to_tuple, convert_to_jsonl
import pandas as pd
import torch
import torch.nn as nn

cwd = os.getcwd()

# ----------------------------- TRANSFORM DATASET ---------------------------- #
# Initialize the tokenizer and the model
tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-large')

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
model = GPT2DoubleHeadsModel.from_pretrained('microsoft/DialoGPT-large')
optimizer = AdamW(model.parameters())

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

# Set the learning rate and the warmup steps
learning_rate = 1e-4
warmup_steps = 1000

# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1)

# ------------------------------- TRAINING LOOP ------------------------------ #
# Train the model
print('Training the model...')
for epoch in range(num_epochs):
    # Initialize the total loss and the number of batches
    total_loss = 0
    num_batches = 0

    # Iterate over the batches
    for i in range(0, len(input_tensors), batch_size):
        # Get the input and output batches
        input_batch = input_tensors[i:i + batch_size]
        output_batch = output_tensors[i:i + batch_size]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_batch, labels=output_batch)
        loss, logits = outputs[:2]

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update the total loss and the number of batches
        total_loss += loss.item()
        num_batches += 1

    # Compute the average loss
    avg_loss = total_loss / num_batches

    # Print the loss
    print(f'Epoch: {epoch + 1}, Loss: {avg_loss:.4f}')

# Save the model
model.save_pretrained(cwd + '/models/dialogpt_model')
    

