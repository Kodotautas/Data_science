import torch

# Define a function to generate a response
def chatbot_response(user_input, model, tokenizer, device):
    """function generates response from user input
    Args:
        user_input (string): user input
        model (): model to generate response
        tokenizer (): tokenizer to tokenize user input
        device (): device to run model on (cpu or gpu)
    Returns:
        _type_: _description_
    """    
    # Tokenize the user input
    input_ids = tokenizer.encode(user_input, return_tensors='pt').to(device)

    # Forward pass
    output = model.generate(input_ids, max_length=100, do_sample=True)

    # Decode the output and remove the batch dimension
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response