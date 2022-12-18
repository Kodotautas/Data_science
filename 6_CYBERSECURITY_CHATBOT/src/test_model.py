import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = os.getcwd()
# one level up
path = os.path.dirname(path)

device = torch.device("cpu")


# ------------------------------- MODEL TESTING ------------------------------ #
# Load the trained model
model = AutoModelForCausalLM.from_pretrained(path + '/models/chatbot_model')
model.to(device)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(path + '/models/chatbot_tokenizer')

# Test the chatbot with a question
question = 'Hello, how are you doing today?'

# Tokenize the question
input_ids = tokenizer.encode(question, return_tensors='pt').to(device)

# Generate the answer
chat_history_ids = model.generate(input_ids, max_length=100, pad_token_id=50256)

# Decode the answer
answer = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
print(answer)
