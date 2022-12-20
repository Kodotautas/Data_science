#script load pre-trained model and test in with user input
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = os.getcwd()
# one level up
path = os.path.dirname(path)


# -------------------------------- LOAD MODEL -------------------------------- #
# Load the tokenizer and model from models/dialogpt folder
tokenizer = AutoTokenizer.from_pretrained(path + '/models/dialogpt')
model = AutoModelForCausalLM.from_pretrained(path + '/models/dialogpt')

device = torch.device("cpu")

# Create a loop to prompt the user for input and generate a response
while True:
     # Get the input from the user
     input_ = input("User: ")
     # Encode the input
     input_ids = tokenizer.encode(input_, return_tensors='pt')
     # Generate the response
     output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
     # Decode the response
     print("Bot: ", tokenizer.decode(output[0], skip_special_tokens=True))