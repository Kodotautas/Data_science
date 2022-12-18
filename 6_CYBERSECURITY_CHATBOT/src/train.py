# Description: Train the chatbot model with the training data and save the model.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_preprocess import loss_function

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-large')

# Set the device to run on (CPU or GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)

# Load the training data
conversations = [['Hello, how are you doing today?', 'I am doing well, thank you. How about you?'],
    ['I am doing well too. Do you have any plans for the weekend?', 'Not really, I was thinking of just relaxing at home.'],
    ['That sounds like a good plan. I might join you.', 'That would be great! We can watch movies and cook together.']
    ]

# Preprocess the training data
input_ids = []
attention_masks = []
for conversation in conversations:
    # Tokenize the conversation
    input_ids.append(tokenizer.encode(conversation[0], return_tensors='pt').to(device))
    attention_masks.append(torch.ones(input_ids[-1].shape[1], dtype=torch.long, device=device))

# Define the training loop
optimizer = torch.optim.Adam(model.parameters())

num_epochs = 1

for epoch in range(num_epochs):
    print('Training data...')
    # Loop over the training data
    for conversation, input_id, attention_mask in zip(conversations, input_ids, attention_masks):
        # Tokenize the second sentence in the conversation
        target_id = tokenizer.encode(conversation[1], return_tensors='pt').to(device)

        # Allign tensors
        input_id = input_id[:, :target_id.shape[1]]
        target_id = target_id[:, :input_id.shape[1]]
        attention_mask = attention_mask[:input_id.shape[1]]

        # Forward pass
        output = model(input_ids=input_id, attention_mask=attention_mask, labels=target_id)
        loss = output.loss

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Print the loss
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

# Save the model
model.save_pretrained('models/dialogpt_large')

# Test the chatbot
print('Testing the chatbot...')
input_text = 'Hello, how are you doing today?'
input_id = tokenizer.encode(input_text, return_tensors='pt').to(device)
attention_mask = torch.ones(input_id.shape[1], dtype=torch.long, device=device)
response = model.generate(input_ids=input_id, attention_mask=attention_mask, max_length=1024)  # modified here to use input_ids instead of input_id
response_text = tokenizer.decode
