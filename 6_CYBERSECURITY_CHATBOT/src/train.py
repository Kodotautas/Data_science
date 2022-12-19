import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_preprocess import df_to_list, calculate_accuracy

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
conversations = df_to_list(df)
# print(conversations)

# sample training data
# conversations = [['Hello, how are you doing today?', 'I am doing well, thank you. How about you?'],
#     ['I am doing well too. Do you have any plans for the weekend?', 'Not really, I was thinking of just relaxing at home.'],
#     ['That sounds like a good plan. I might join you.', 'That would be great! We can watch movies and cook together.']
#     ]

# Preprocess the training data
input_tensors = []
target_tensors = []
for conversation in conversations:
  input_tensor = tokenizer.encode(conversation[0], return_tensors='pt')
  target_tensor = tokenizer.encode(conversation[1], return_tensors='pt')
  input_tensors.append(input_tensor)
  target_tensors.append(target_tensor)

# Align the tensors
max_len = max(len(input_tensor[0]), len(target_tensor[0]))
for i in range(len(input_tensors)):
  input_tensors[i] = torch.nn.functional.pad(input_tensors[i], (0,max_len-len(input_tensors[i][0])))
  target_tensors[i] = torch.nn.functional.pad(target_tensors[i], (0,max_len-len(target_tensors[i][0])))

# Move the tensors to the device
input_tensors = [tensor.to(device) for tensor in input_tensors]
target_tensors = [tensor.to(device) for tensor in target_tensors]

# Define the optimizer and criterion
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

# test tensors
print(input_tensors[0])
print(target_tensors[0])

# visualize tensors with matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.imshow(input_tensors[0].detach().numpy())
plt.show()


# ------------------------------ TRAIN MODEL ------------------------------ #
# Train the model
print('Training the model...')
epochs = 10
for epoch in range(epochs):
  for i in range(len(input_tensors)):
    # Get the input and target tensors
    input_tensor = input_tensors[i]
    target_tensor = target_tensors[i]

    # Forward pass
    logits = model(input_tensor, labels=target_tensor)[1]

    # Calculate the loss
    loss = criterion(logits.view(-1, logits.shape[-1]), target_tensor.view(-1))

    # Backward pass
    loss.backward()

    # Update the parameters
    optimizer.step()

    # Zero the gradients
    optimizer.zero_grad()

  # Print the loss and accuracy
  print('Epoch: {}/{}'.format(epoch+1, epochs))
  print('Loss: {}'.format(loss.item()))
  print('Accuracy: {}'.format(calculate_accuracy(logits, target_tensor)))
  
# Save the model
model.save_pretrained(path + '/models/dialogpt')

# Save the tokenizer
tokenizer.save_pretrained(path + '/models/dialogpt')


# ------------------------------ TEST MODEL ------------------------------ #
# Test the model
input_tensor = tokenizer.encode('Hello', return_tensors='pt')
input_tensor = torch.nn.functional.pad(input_tensor, (0,max_len-len(input_tensor[0])))
input_tensor = input_tensor.to(device)
output = model.generate(input_tensor, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
print(tokenizer.decode(output[0], skip_special_tokens=True))
