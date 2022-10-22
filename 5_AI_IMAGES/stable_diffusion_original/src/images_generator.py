import os
from importlib.resources import path
from pyparsing import Word
from regex import W
import config
from huggingface_hub import notebook_login
import torch
import gc
from torch import autocast, batch_norm
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
    image.save(path + '/outputs/raw_images/' + prompt[0:30] + '.png')
    return print("Image saved!")