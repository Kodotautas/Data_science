import os
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

import warnings
warnings.filterwarnings('ignore')

path = "C:/Users/Kodotautas/Desktop/Data_science/5_AI_IMAGES/stable_diffusion_original"


# -------------------------------- LOAD MODEL -------------------------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)


# -------------------------------- GENERATE IMAGES -------------------------------- #
def tune_image(image_path):
    """function enlarges image and saves it to the same folder
    Args:
        image_path (_type_): path to the image
    """    
    #get image name
    image_name = os.path.basename(image_path)
    image = Image.open(image_path).convert('RGB')
    sr_image = model.predict(image)
    sr_image.save(path + f'/outputs/upscaled_images/{image_name}.png')
    print(f'{image_name} enlarged and saved to disk.')

#find all images in the folder


#tune images from folder
for image in os.listdir(path + '/outputs/filtered/filter/'):
    tune_image(path + f'/outputs/filtered/filter/{image}')

print('All Done!')














# import os
# from importlib_metadata import version
# import replicate
# from sklearn.preprocessing import scale

# path = "C:/Users/Kodotautas/Desktop/Data_science/5_AI_IMAGES/stable_diffusion_original"

# client = replicate.Client(api_token='4118258dc15c24ee723a9c548737c919042133d2')


# # -------------------------------- LOAD MODEL -------------------------------- #
# print("Loading model...")
# model = client.models.get("xinntao/realesrgan")


# # ---------------------------------- PREDICT --------------------------------- #
# print("Enlarging image...")
# output = model.predict(image='C:/Users/Kodotautas/Desktop/Data_science/5_AI_IMAGES/stable_diffusion_original/outputs/raw_images/superman.png',
# version='General - RealESRGANplus', scale=4, face_enchance=True)