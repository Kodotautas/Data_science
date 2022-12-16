from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize the DialoGPT tokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')

# Initialize the DialoGPT model
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-large')

# Define the training data
data = [
    {"input": "Why do I need to worry about Cyber Security?", "output": "Cyber Security protects unauthorized access and or criminal use of your data."},
    {"input": "Should I have a different password for every website?", "output": "Yes. If you use the same password on every web- site and someone gets access to it."},
    {"input": "How can I remember all of my passwords?", "output": "You can use a password manager. There are free options and low cost ones."},
]

# Encode the training data
encoded_data = []
for pair in data:
    input_text = pair['input']
    output_text = pair['output']
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output_ids = tokenizer.encode(output_text, return_tensors='pt')
    encoded_data.append({'input_ids': input_ids, 'output_ids': output_ids})


# Set the number of training epochs
num_epochs = 10

# Set the learning rate
lr = 1e-4

# Set the device to use for training
device = 'cpu'

# Set the model to train mode
model.train()

# Use Adam optimizer for training
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Start training loop
for epoch in range(num_epochs):
    for data in encoded_data:
        # Get the input and output tensors
        input_ids = data['input_ids'].to(device)
        output_ids = data['output_ids'].to(device)
        print(input_ids.shape)

        # Reset the gradients
        optimizer.zero_grad()

        # Forward pass
        loss = model(input_ids, labels=output_ids)

        # Backward pass and optimization
        loss.backward()

        # Update the parameters
        optimizer.step()
        
    # Print the loss
    print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')

# Save the model
model.save_pretrained('models/dialogpt')