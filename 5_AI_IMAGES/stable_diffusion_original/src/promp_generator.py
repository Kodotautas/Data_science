import glob
import random
from importlib.resources import path
from transformers import AutoTokenizer, AutoModelForCausalLM

import warnings
warnings.filterwarnings("ignore")

path = "C:/Users/Kodotautas/Desktop/Data_science/5_AI_IMAGES/stable_diffusion_original"


# -------------------------------- LOAD MODEL -------------------------------- #
tokenizer = AutoTokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
model = AutoModelForCausalLM.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")


# -------------------------------- GET INPUTS ------------------------------- #
def choose_word():
    #merge txt files into one
    with open(path + "/material/words.txt", "w") as outfile:
        for filename in glob.glob(path + "/material/*.txt"):
            with open(filename, "r") as infile:
                outfile.write(infile.read())

    #read output file
    with open(path + "/material/words.txt", "r") as f:
        words = f.read().splitlines()

    #choose random item from list
    word = random.choice(words)
    return word


# -------------------------------- GENERATE PROMPT -------------------------------- #
def generate_prompts(word):
    prompt = word
    #tokenize
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    #generate
    output = model.generate(input_ids, max_length=1000, do_sample=True, top_k=50, top_p=0.95, temperature=0.9, num_return_sequences=1)
    #decode
    prompt = tokenizer.decode(output[0], skip_special_tokens=True)
    return prompt