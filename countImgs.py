import json
import os

count = 0
scraped_data_folder = 'scrapedData'

for filename in os.listdir(scraped_data_folder):
    if filename.endswith('.json'):
        file_path = os.path.join(scraped_data_folder, filename)
        with open(file_path) as f:
            data = json.load(f)
            for key in data:
                count += len(data[key])
        print(f"Total number of URLs in {filename}", count)

print("Total number of URLs: ", count)
