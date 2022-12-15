import torch
import torch.nn as nn
import torch.optim as optim

# Define the model architecture
class ChatBot(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(ChatBot, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input):
        # input: (batch_size, seq_len)
        embedded = self.embedding(input)  # embedded: (batch_size, seq_len, embedding_dim)
        lstm_output, (hidden, cell) = self.lstm(embedded)  # lstm_output: (batch_size, seq_len, hidden_dim)
        linear_output = self.linear(lstm_output)  # linear_output: (batch_size, seq_len, vocab_size)
        return linear_output