import os
from PIL import Image
import hashlib

dataset_folder = './dataset'

imageTypes = [".jpg", ".jpeg", ".png"]

image_count = 0
duplicate_count = 0
hash_set = set()

for root, dirs, files in os.walk(dataset_folder):
    for filename in files:
        if filename.endswith(tuple(imageTypes)):
            try:
                img_path = os.path.join(root, filename)
                with open(img_path, "rb") as f:
                    img_hash = hashlib.sha256(f.read()).hexdigest()
                if img_hash not in hash_set:
                    hash_set.add(img_hash)
                    img = Image.open(img_path)
                    img.verify()
                    image_count += 1
                    print(f"Checked {image_count} images", end='\r')
                else:
                    print(f"Duplicate image found and removed: {filename}")
                    os.remove(img_path)
                    duplicate_count += 1
            except (IOError, SyntaxError) as e:
                print('Bad file:', filename)
                os.remove(os.path.join(root, filename))

print(f"Checked {image_count} images and removed {duplicate_count} duplicate images")
