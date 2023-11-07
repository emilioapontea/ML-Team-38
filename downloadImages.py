import json
import os
import urllib.request

# Load the JSON file
with open('nikePrices.json') as f:
    data = json.load(f)

# Iterate over the price categories
for price, urls in data.items():
    # Create a directory for each price category
    os.makedirs(os.path.join("./dataset", price), exist_ok=True)
    
    # Iterate over the URLs in each price category
    for url in urls:
        # Extract the UUID from the URL
        image_uuid = url.split('t_product_v1/')[-1].split("/")[0].split(",")[0]
        
        # Download the image and save it in the corresponding directory
        urllib.request.urlretrieve(url, os.path.join("./dataset", price, f'{image_uuid}.jpg'))
