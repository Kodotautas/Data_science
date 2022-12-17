import torch

# Define a function to generate chatbot responses
def chatbot_response(prompt, tokenizer, model, device):
  # Tokenize the prompt and add the special [CLS] and [SEP] tokens
  tokenized_prompt = tokenizer.encode(prompt, add_special_tokens=True)

  # Convert the tokenized prompt to a tensor and move it to the device
  prompt_tensor = torch.tensor(tokenized_prompt).unsqueeze(0).to(device)

  # Generate the chatbot response
  chatbot_response = model.generate(prompt_tensor, max_length=50, pad_token_id=0, top_p=0.9, top_k=10)[0]

  # Convert the tokenized response to a string
  response = tokenizer.decode(chatbot_response, skip_special_tokens=True)

  return response