import os
from PIL import Image

dataset_folder = './dataset'

imageTypes = [".jpg", ".jpeg", ".png"]

image_count = 0

for filename in os.listdir(dataset_folder):
    if filename.endswith(tuple(imageTypes)):
        try:
            img = Image.open(os.path.join(dataset_folder, filename)) 
            img.verify()
            image_count += 1
            print(f"Checked {image_count} images")
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename)
            os.remove(os.path.join(dataset_folder, filename))
