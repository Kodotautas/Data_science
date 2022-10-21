import os
from importlib.resources import path
from click import prompt
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



# -------------------------------- GENERATE IMAGES -------------------------------- #
#text input
prompt = "Professor X, wearing a top hat, short shorts, muscular, D&D, fantasy, intricate, elegant, highly detailed, digital painting, artstation, concept art, matte, sharp focus, illustration, art by Artgerm and Greg Rutkowski and Alphonse Mucha"

print("Generating image...")
image = pipe(prompt).images[0] 

#save image
image.save(path + '/outputs/raw_images/' + prompt[0:30] + '.png')

print("Image saved!")