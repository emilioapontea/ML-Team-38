import json
import os
import urllib.request
from urllib.parse import quote

# Load the JSON file
with open('./scrapedData/adidasPrices.json') as f:
    data = json.load(f)

# Iterate over the price categories
for price, urls in data.items():
    # Create a directory for each price category
    os.makedirs(os.path.join("./dataset", price), exist_ok=True)
    
    # Iterate over the URLs in each price category
    for url in urls:
        # Extract the UUID from the URL
        image_uuid = url.split('images/')[1].split("/")[0]
        # print(image_uuid)
        
        # Define the path where the image will be saved
        image_path = os.path.join("./dataset", price, f'{image_uuid}.jpg')
        
        # Check if the image already exists
        if not os.path.exists(image_path):
            # Download the image and save it in the corresponding directory
            print(f"Downloaded {url}")
            encoded_url = quote(url, safe="%/:=&?~#+!$,;'@()*[]")
            urllib.request.urlretrieve(encoded_url, image_path)
        else:
            print("Image already downloaded!")