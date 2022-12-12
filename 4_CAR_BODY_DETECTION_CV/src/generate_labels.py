# generate labels for computer vision training
import os

cwd = os.getcwd()
path = os.path.dirname(cwd)
print(path)

labels_list = ['Buggy', 'Convertible', 'Coupe', 'Hatchback', 'Limousine', 'MiniVan', 'Sedan']

# balance dataset main function
def balance_dataset(path, label, max_images):
    # path: path to the images folder
    # label: label to use for the images
    # max_images: maximum number of images to keep
    images = os.listdir(path + label)
    if len(images) > max_images:
            for image in images[max_images:]:
               os.remove(path + label + '/' + image)

# lauch balance dataset for subfolders
for label in labels_list:
    print('Balancing dataset for: ' + label)
    balance_dataset(f'{path}/data/images/', label, 250)

# create txt file with labels where each line is a label and image name
def create_labels_file(path, label):
    images = os.listdir(path + '/data/images/' + label)
    with open(f'{path}/src/labels.txt', 'a') as f:
        for image in images:
            f.write('\n' + image + ',' + label)

# lauch create labels file for subfolders
for label in labels_list:
    print('Creating labels file for: ' + label)
    create_labels_file(path, label)