import requests
from bs4 import BeautifulSoup
import json
import os

reebokPrices = {}

reebokFile = "./scrapedData/reebokPrices.json"

if os.path.exists(reebokFile) and os.path.getsize(reebokFile) > 0:
    with open(reebokFile, "r") as jsonFile:
        reebokPrices = json.load(jsonFile)
else:
    print("File is empty or does not exist")

# print(reebokPrices)

def gatherRowData(jsonData, prevData=reebokPrices):
    # print(jsonData['props']['initialProps']['pageProps']['initialApolloState']['ROOT_QUERY'])
    # print(jsonData["props"]["initialProps"]["pageProps"]["initialApolloState"]["ROOT_QUERY"]['productSearch({"bruid":"","category":"600000057","deviceType":"desktop","facets":[],"fetchType":"bloomreach","keyword":null,"limit":49,"locationCode":null,"offset":0,"sortBy":null,"sortOrder":null})']['results'])
    # for each in jsonData["props"]["initialProps"]["pageProps"]["initialApolloState"]["ROOT_QUERY"]['productSearch({"bruid":"","category":"600000057","deviceType":"desktop","facets":[],"fetchType":"bloomreach","keyword":null,"limit":49,"locationCode":null,"offset":0,"sortBy":null,"sortOrder":null})']["results"]:
    root_query_keys = list(jsonData["props"]["initialProps"]["pageProps"]["initialApolloState"]["ROOT_QUERY"].keys())
    second_key = root_query_keys[1]
    for each in jsonData["props"]["initialProps"]["pageProps"]["initialApolloState"]["ROOT_QUERY"][second_key]["results"]:
        price = str(each["reqularPrice"])
        portraitURL = each["thumbImage"].split("?io=transform:scale,width:280,height:280")[0]
        if "variants" in each:
            for variant in each["variants"]:
                variant_price = str(variant["regularPrice"])
                variant_portraitURL = variant["variant_thumb_image"].split("?io=transform:scale,width:280,height:280")[0]
                if variant_price not in prevData.keys():
                    prevData[variant_price] = [variant_portraitURL]
                elif variant_portraitURL not in prevData[variant_price]:
                    prevData[variant_price].append(variant_portraitURL)
        if price not in prevData.keys():
            prevData[price] = [portraitURL]
        elif portraitURL not in prevData[price]:
            prevData[price].append(portraitURL)

def collectData(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')
    script_tag = soup.find('script', {'id': '__NEXT_DATA__'})
    data = json.loads(script_tag.string)
    gatherRowData(jsonData=data)

    with open(reebokFile, "w") as jsonFile:
        json.dump(reebokPrices, jsonFile, indent=2)

for i in range(1, 22):  # assuming there are 10 pages to scrape
    url = f"https://www.reebok.com/c/600000057/collection-shoes?page={i}"
    collectData(url)

with open(reebokFile, "w") as jsonFile:
    json.dump(reebokPrices, jsonFile, indent=2)

