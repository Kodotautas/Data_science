import os
from importlib.resources import path
from random import random
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
pipe = StableDiffusionPipeline.from_pretrained(path + '/models/stable-diffusion-v1-4',
                                                revision="fp16", 
                                                torch_dtype=torch.float16).to("cuda")
print('Model loaded.')


# -------------------------------- GENERATE IMAGES -------------------------------- #
def generate_image(prompt):
    print("Generating image...")
    image = pipe(prompt).images[0]
    #add metadata
    image.metadata = {'prompt': prompt}
    image.metadata = {'model': 'stable-diffusion-v1-4'}
    image.metadata = {'author': 'Vytautas Lukosiunas'}
    #save image
    image.save(path + '/outputs/raw_images/' + prompt.split()[0] 
                                            + prompt.split()[2] 
                                            + prompt.split()[3] 
                                            + '.png')
    #add promt and prompt[0:30] to prompts.txt
    with open(path + '/outputs/prompts/prompts.txt', 'a') as f:
        f.write(prompt[0:30] + '.png' + '\n')
        f.write(prompt + '\n')
        f.write('--------------------------------------------------' + '\n')
        f.close()
    return print("Image & prompt saved!")