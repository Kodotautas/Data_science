# generate labels for computer vision training
import os

path = 'C:/Users/vytlksn/OneDrive - GPC/Desktop/Data_science/4_CAR_BODY_DETECTION_CV'

labels_list = ['Buggy', 'Convertible', 'Coupe', 'Hatchback', 'Limousine', 'MiniVan', 'Sedan']

# balance dataset main function
def balance_dataset(path, label, max_images):
    # path: path to the images folder
    # label: label to use for the images
    # max_images: maximum number of images to keep
    images = os.listdir(path + '/data/images/' + label)
    if len(images) > max_images:
            for image in images[max_images:]:
               os.remove(path + '/data/images/' + label + '/' + image)

# lauch balance dataset for subfolders
for label in labels_list:
    print('Balancing dataset for: ' + label)
    balance_dataset(path + '/data/images/', label, 270)


# other paths
images_path = path + '/data/images'
output_file = path + '/data/labels/labels.txt'

def create_labels_file(path, label, output_file):
    # create a labels file for training
    # path: path to the images folder
    # label: label to use for the images
    # output_file: path to the output file
    with open(output_file, 'a') as f:
        for filename in os.listdir(path):
            f.write(os.path.join(path, filename) + ',' + label)

for label in labels_list:
    print('Creating labels for: ' + label)
    create_labels_file(images_path + '/' + label, label, output_file)

print('All labels created!')