import requests
import json
import os

menURL = "https://api.nike.com/cic/browse/v2?queryid=products&anonymousId=A1400060322AE5F1E7B9A90E5536DC4C&country=us&endpoint=%2Fproduct_feed%2Frollup_threads%2Fv2%3Ffilter%3Dmarketplace(US)%26filter%3Dlanguage(en)%26filter%3DemployeePrice(true)%26filter%3DattributeIds(16633190-45e5-4830-a068-232ac7aea82c%2C0f64ecc7-d624-4e91-b171-b83a03dd8550)%26anchor%3D00%26consumerChannelId%3Dd9a5bc42-4b9c-4976-858a-f159cf99c647%26count%3D48"
womenURL = "https://api.nike.com/cic/browse/v2?queryid=products&anonymousId=A1400060322AE5F1E7B9A90E5536DC4C&country=us&endpoint=%2Fproduct_feed%2Frollup_threads%2Fv2%3Ffilter%3Dmarketplace(US)%26filter%3Dlanguage(en)%26filter%3DemployeePrice(true)%26filter%3DattributeIds(7baf216c-acc6-4452-9e07-39c2ca77ba32%2C16633190-45e5-4830-a068-232ac7aea82c)%26anchor%3D00%26consumerChannelId%3Dd9a5bc42-4b9c-4976-858a-f159cf99c647%26count%3D48"

nikePrices = {}

nikeFile = "../scrapedData/nikePrices.json"

if os.path.exists(nikeFile) and os.path.getsize(nikeFile) > 0:
    with open(nikeFile, "r") as jsonFile:
        nikePrices = json.load(jsonFile)
else:
    print("File is empty or does not exist")

print(nikePrices)

def gatherRowData(jsonData, prevData=nikePrices):
    for each in jsonData:
        price = str(each.get("price").get("fullPrice"))
        portraitURL = each.get("images").get("portraitURL")
        if "images/" in portraitURL and "/t_product_v1" in portraitURL:
            start = portraitURL.find("images/") + len("images/")
            end = portraitURL.find("/t_product_v1")
            portraitURL = portraitURL.replace(portraitURL[start:end+1], "")
            print(portraitURL)
        # print(type(price))
        if price not in prevData.keys():
            prevData[price] = [portraitURL]
        elif portraitURL not in prevData[price]:
            prevData[price].append(portraitURL)  # Append to the existi

def collectData(url):
    req = requests.get(url, headers="")
    data = req.json().get("data").get("products").get("products")
    count = 00
    while (data != None):
        gatherRowData(jsonData=data)

        count += 24
        url = url.replace("anchor%3D{}".format(
            count-24), "anchor%3D{}".format(count))
        print(url)
        req = requests.get(url, headers="")
        data = req.json().get("data").get("products").get("products")

        with open(nikeFile, "w") as jsonFile:
            json.dump(nikePrices, jsonFile, indent=2)


collectData(womenURL)
collectData(menURL)

with open(nikeFile, "w") as jsonFile:
    json.dump(nikePrices, jsonFile, indent=2)
