from transformers import AutoTokenizer, AutoModelForCausalLM

import warnings
warnings.filterwarnings('ignore')

model_name = 'microsoft/DialoGPT-large'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# test model with input
user_input = "Hello, how are you? Where are you from?"

# encode the new user message to be used by our model
input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
chat_history_ids = model.generate(
    input_ids,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    max_length=1000,
    top_k=100,
    top_p=0.95,
    temperature=0.8,
)

# append the encoded message to the past history so the model is aware of past context
decoded_message = tokenizer.decode(
    chat_history_ids[:, input_ids.shape[-1]:][0],
    skip_special_tokens=True
)

print(decoded_message)
