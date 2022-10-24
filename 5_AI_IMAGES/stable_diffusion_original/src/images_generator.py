import os
import string
import random
import numpy as np
from importlib.resources import path
from pyparsing import Word
from regex import W
import config
from huggingface_hub import notebook_login
import torch
import gc
from torch import autocast, batch_norm, randint
from diffusers import StableDiffusionPipeline

torch.cuda.empty_cache()
gc.collect()

path = "C:/Users/Kodotautas/Desktop/Data_science/5_AI_IMAGES/stable_diffusion_original"


# -------------------------------- LOAD MODEL -------------------------------- #
print("Loading model...")
pipe = StableDiffusionPipeline.from_pretrained(path + '/models/stable-diffusion-v1-5',
                                                revision="fp16", 
                                                torch_dtype=torch.float16).to("cuda")
print('Model loaded.')


# -------------------------------- GENERATE IMAGES -------------------------------- #
def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def generate_image(prompt):
    print("Generating image...")
    image = pipe(prompt).images[0]
    #add metadata
    image.metadata = {'prompt': prompt}
    image.metadata = {'model': 'stable-diffusion-v1-4'}
    image.metadata = {'author': 'Vytautas Lukosiunas'}
    #save image
    name = prompt.split()[0] + '_' + get_random_string(10)
    image.save(path + '/outputs/raw_images/' + name + '.png')
    #add promt and prompt[0:30] to prompts.txt
    with open(path + '/outputs/prompts/prompts.txt', 'a') as f:
        f.write(name  + '.png' + '\n')
        #remove non ascii characters
        prompt = ''.join([i if ord(i) < 128 else ' ' for i in prompt])
        f.write(prompt + '\n')
        f.write('--------------------------------------------------' + '\n')
        f.close()
    return print("Image & prompt saved!")