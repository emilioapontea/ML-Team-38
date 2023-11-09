import requests
from bs4 import BeautifulSoup
import json
import os

pumaPrices = {}

pumaFile = "./scrapedData/pumaPrices.json"

if os.path.exists(pumaFile) and os.path.getsize(pumaFile) > 0:
    with open(pumaFile, "r") as jsonFile:
        pumaPrices = json.load(jsonFile)
else:
    print("File is empty or does not exist")

def gatherRowData(jsonData, prevData=pumaPrices):
    jsonData = jsonData["data"]
    jsonData = list(jsonData["categoryByUrl"]["products"]["nodes"])
    for each in jsonData:
        price = str(each["variantProduct"]["price"])
        imageURLs = [(color["image"]["href"]).split("PNA")[0] for color in each["masterProduct"]["colors"]]
        for imageURL in imageURLs:
            if price not in prevData.keys():
                prevData[price] = [imageURL]
            elif imageURL not in prevData[price]:
                prevData[price].append(imageURL)

def gatherRowDataJSON(jsonData, prevData=pumaPrices):
    jsonData = json.loads(jsonData["props"]["urqlState"]["-12898589755"]["data"])
    jsonData = list(jsonData["categoryByUrl"]["products"]["nodes"])
    for each in jsonData:
        price = str(each["variantProduct"]["price"])
        imageURLs = [(color["image"]["href"]).split("PNA")[0] for color in each["masterProduct"]["colors"]]
        for imageURL in imageURLs:
            if price not in prevData.keys():
                prevData[price] = [imageURL]
            elif imageURL not in prevData[price]:
                prevData[price].append(imageURL)

def collectData(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')
    script_tag = soup.find('script', {'id': '__NEXT_DATA__'})
    data = json.loads(script_tag.string)
    gatherRowData(jsonData=data)

    with open(pumaFile, "w") as jsonFile:
        json.dump(pumaPrices, jsonFile, indent=2)

def collectDataFromJSON(filename):
    data = json.loads(filename)
    gatherRowData(jsonData=data)

    with open(pumaFile, "w") as jsonFile:
        json.dump(pumaPrices, jsonFile, indent=2)

url = f"https://us.puma.com/us/en/men/shoes?offset=500"
collectData(url)


