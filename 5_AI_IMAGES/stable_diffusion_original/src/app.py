from zmq import Errno
import promp_generator
import images_generator
# import images_tuner

import warnings
warnings.filterwarnings('ignore')

#number of images to generate
n = 1000


# -------------------------------- APPLICATION ------------------------------- #
def main():
    #generate main word and prompt
    # word = promp_generator.choose_word()
    prompt = promp_generator.generate_prompts('Cute, big smile punk animal, clean background') #CLIP allows max 77 characters
    # print(prompt)
    #generate image
    images_generator.generate_image(prompt)
    return print("All Done!")

if __name__ == "__main__":
    for i in range(n):
        print("%s/%s" % (i+1, n))
        main()