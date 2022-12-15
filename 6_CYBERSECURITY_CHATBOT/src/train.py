from models.model import ChatBot
import torch
import torch.nn as nn
import torch.optim as optim
import random

def train(model, optimizer, train_data, vocab_size, embedding_dim, hidden_dim, learning_rate, batch_size, num_epochs):
    # Train the model
    for epoch in range(num_epochs):
        # Shuffle the training data
        random.shuffle(train_data)
      
        # Split the training data into batches
        num_batches = len(train_data) // batch_size
        for i in range(num_batches):
            # Get the current batch
            batch = train_data[i*batch_size : (i+1)*batch_size]
          
            # Create the input and target tensors for the current batch
            # and move them to the GPU (if available)
            input_tensors = torch.tensor([example[0] for example in batch])
            target_tensors = torch.tensor([example[1] for example in batch])
            if torch.cuda.is_available():
                input_tensors = input_tensors.cuda()
                target_tensors = target_tensors.cuda()

            # Forward pass
            output = model(input_tensors)

            # Compute the loss
            loss = nn.CrossEntropyLoss()(output, target_tensors)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update the parameters
            optimizer.step()

        # Print the loss after every epoch
        print('Epoch: {}/{}, Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    # Save the model
    torch.save(model.state_dict(), 'models/model.pth')
