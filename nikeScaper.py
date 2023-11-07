import requests
import json

menURL = "https://api.nike.com/cic/browse/v2?queryid=products&anonymousId=A1400060322AE5F1E7B9A90E5536DC4C&country=us&endpoint=%2Fproduct_feed%2Frollup_threads%2Fv2%3Ffilter%3Dmarketplace(US)%26filter%3Dlanguage(en)%26filter%3DemployeePrice(true)%26filter%3DattributeIds(16633190-45e5-4830-a068-232ac7aea82c%2C0f64ecc7-d624-4e91-b171-b83a03dd8550)%26anchor%3D00%26consumerChannelId%3Dd9a5bc42-4b9c-4976-858a-f159cf99c647%26count%3D48"
womenURL = "https://api.nike.com/cic/browse/v2?queryid=products&anonymousId=A1400060322AE5F1E7B9A90E5536DC4C&country=us&endpoint=%2Fproduct_feed%2Frollup_threads%2Fv2%3Ffilter%3Dmarketplace(US)%26filter%3Dlanguage(en)%26filter%3DemployeePrice(true)%26filter%3DattributeIds(7baf216c-acc6-4452-9e07-39c2ca77ba32%2C16633190-45e5-4830-a068-232ac7aea82c)%26anchor%3D00%26consumerChannelId%3Dd9a5bc42-4b9c-4976-858a-f159cf99c647%26count%3D48"

nikePrices = {}

def gatherRowData(jsonData):
    for each in jsonData:
        price = each.get("price").get("fullPrice")
        portraitURL = each.get("images").get("portraitURL")

        if price in nikePrices:
            print(portraitURL)
            nikePrices[price].append(portraitURL)
        else:
            nikePrices[price] = [portraitURL]

def collectData(url):
    req = requests.get(url, headers="")
    data = req.json().get("data").get("products").get("products")
    count = 00
    while (data != None):
        gatherRowData(jsonData=data)

        count += 24
        url = url.replace("anchor%3D{}".format(
            count-24), "anchor%3D{}".format(count))
        req = requests.get(url, headers="")
        data = req.json().get("data").get("products").get("products")

collectData(womenURL)
collectData(menURL)

with open("nikePrices.json", "w") as jsonFile:
    json.dump(nikePrices, jsonFile)
