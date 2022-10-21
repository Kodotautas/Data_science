from importlib.resources import path
from transformers import AutoTokenizer, AutoModelForCausalLM


path = "C:/Users/Kodotautas/Desktop/Data_science/5_AI_IMAGES/stable_diffusion_original"

# -------------------------------- LOAD MODEL -------------------------------- #
tokenizer = AutoTokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
model = AutoModelForCausalLM.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")


# -------------------------------- GENERATE PROMPTS -------------------------------- #
#input
prompt = "Professor X"

#tokenize
input_ids = tokenizer.encode(prompt, return_tensors='pt')

#generate
output = model.generate(input_ids, max_length=1000, do_sample=True, top_k=50, top_p=0.95, temperature=0.9, num_return_sequences=1)

#decode
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)