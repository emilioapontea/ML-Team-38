import os
from PIL import Image

dataset_folder = 'path_to_your_dataset_folder'

imageTypes = [".jpg", ".jpeg", ".png"]

for filename in os.listdir(dataset_folder):
    if filename.endswith(tuple(imageTypes)):
        try:
            img = Image.open(os.path.join(dataset_folder, filename)) 
            img.verify()
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename)
            os.remove(os.path.join(dataset_folder, filename))
