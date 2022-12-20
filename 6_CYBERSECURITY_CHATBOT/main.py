#script load pre-trained model and test in with user input
import os
import torch

path = os.getcwd()


# -------------------------------- LOAD MODEL -------------------------------- #
# Load the tokenizer and model
tokenizer = torch.load(path + '/models/tokenizer.pt')
model = torch.load(path + '/models/model.pt')

# Set the device to run on (CPU or GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model.to(device)

# ------------------------------- TEST THE MODEL ----------------------------- #
# conversation with the bot
print('Welcome to the Cybersecurity FAQ chatbot. Type "quit" to exit.')
while True:
     # get user input
     user_input = input('Q: ')
     # check if user wants to quit
     if user_input.lower() == 'quit':
          print('Bye!')
          break
     # encode user input
     encoded = tokenizer.encode(user_input, add_special_tokens=True, pad_to_max_length=True, truncation=True, max_length=512)
     # convert to tensor
     input_ids = torch.tensor(encoded).unsqueeze(0).to(device)
     # generate response
     response = model.generate(input_ids, max_length=512, pad_token_id=tokenizer.eos_token_id)
     # decode response
     print('A:', tokenizer.decode(response[0], skip_special_tokens=True))
