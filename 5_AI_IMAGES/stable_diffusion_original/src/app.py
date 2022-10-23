import promp_generator
import images_generator
# import images_tuner

import warnings
warnings.filterwarnings('ignore')

#number of images to generate
n = 20

# -------------------------------- APPLICATION ------------------------------- #
def main():
    #generate main word and prompt
    # word = promp_generator.choose_word()
    prompt = promp_generator.generate_prompts('scene of')
    print(prompt)
    #generate image
    images_generator.generate_image(prompt)
    return print("All Done!")


if __name__ == "__main__":
    for i in range(n):
        print("%s/%s" % (i+1, n))
        main()