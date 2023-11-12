import requests
from bs4 import BeautifulSoup
import json
import os

# view-source:https://www.asics.com/us/en-us/mens-shoes/c/aa10200000/?start=600&sz=700
# view-source:https://www.asics.com/us/en-us/mens-shoes/c/aa10200000/?start=700&sz=900

asicsPrices = {}

asicsFile = "./scrapedData/asicsPrices.json"

if os.path.exists(asicsFile) and os.path.getsize(asicsFile) > 0:
    with open(asicsFile, "r") as jsonFile:
        asicsPrices = json.load(jsonFile)
else:
    print("File is empty or does not exist")

def gatherRowData(li_tags, prevData=asicsPrices):
    for li in li_tags:
        try:
            price = li.find("span", class_="price-standard outlet-pricing").text.strip("$").split(".")[0]
            image = li.find("img", class_="b-lazy js-tile-img product-tile__image")["data-src"].replace("$productlist$", "$zoom$")
            print(price)
            print(image)
            if price not in prevData.keys():
                prevData[price] = [image]
            elif image not in prevData[price]:
                prevData[price].append(image)
        except AttributeError:
            with open(asicsFile, "w") as jsonFile:
                json.dump(asicsPrices, jsonFile, indent=2)
            continue

def collectData(url):
    # print(url.split(".com")[1])
    # headers = {
    #     "authority": "www.asics.com",
    #     "method": "GET",
    #     "path": str(url.split(".com")[1]),
    #     "scheme": "https",
    #     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    #     "Accept-Encoding": "gzip, deflate, br",
    #     "Accept-Language": "en-US,en;q=0.5",
    #     "Cache-Control": "max-age=0",
    #     "Cookie": "t_user_type=new; dwac_2a962b7bf6f16de6b195c501eb=8Ah-lZJkFltCz7cQEWqU9XR0l0mAT8ynIzM%3D|dw-only|||USD|false|US%2FEastern|true; cqcid=cdhhLlXSl2atOBF1RocbeGJGtW; cquid=||; sid=8Ah-lZJkFltCz7cQEWqU9XR0l0mAT8ynIzM; dwanonymous_a0bee6d5d02f73f6e619681126569802=cdhhLlXSl2atOBF1RocbeGJGtW; __cq_dnt=0; dw_dnt=0; dwsid=w1vzbr9GeY3JRMI7KVHLTEcnGrKtbaNxqLhXRHc65tgyT-847aVM9mqDgxzVv2pJA4IJWGp-zZ4IauU-ImqfMQ==; user_country=US; bm_sz=74A8C6ED16DB048858D099F3F14E4D87~YAAQ0pfAF84E0amLAQAAAJHRthX1ci2ltfqqisbQbuwRHrSoOYj6ECYTefsmRMVDFx8/vmf9Q5sfD08MlmVeZV1ZsYin3xwmjvb7wGLM9ubdsgzGZBemNhVwu6AePWjmBsX/Ieti65bxc0QvC7RZjZ0CaEpxa653AgyHAbv16cNs6XGUY8RL31zCu6jTvcwW1Y770gFL9jvesCMwfiH1AO2ra2tOXWP0QuTVhEuRSY4ZWzDG6vEQO3oJh0chymuWRKI2lwSWxGUBK0kkN7HLsjZ/SCGs+K8GXiu81+rIKGNqqw==~3487043~3359302; CONSENTMGR=c1:1%7Cc2:1%7Cc3:1%7Cc4:1%7Cc5:1%7Cc6:1%7Cc7:1%7Cc8:1%7Cc9:1%7Cc10:1%7Cc11:1%7Cc12:1%7Cc13:1%7Cc14:1%7Cc15:1%7Cts:1699579270470%7Cconsent:true; utag_main__sn=1; utag_main_ses_id=1699579270474%3Bexp-session; utag_main_dcsyncran=1%3Bexp-session; dw=1; dw_cookies_accepted=1; ak_bmsc=EAF0BAC5579FEDA5E7E6A56021801E85~000000000000000000000000000000~YAAQ0pfAF+4E0amLAQAAaZjRthU23yZyM47jdCulv4g34snVefmGaA2fxzSbVetg/L75aYvYNayV0lEo4E829PToSLm49Zb3y+Kuww0ydiLwa1mUO5q+M4SRSZHtXAIhnYl6CJtDwFDKWQZyW2U6OuT9+xW05QU2nlCMLEtvPoL/Q+1nIe8HvKtTQuYSZtUhBUnz7svDwR8FUVvvL3Za8SSYSOjXnDK0CkN7/bmiaCfonH1xJdd/dMT0yj8x2CQ9UBobhW7c9QVPEkaI0rDI+Zs/0XLIPwwfhiRCtXKlTZebLpXTW6qIMQm89A8GIded+H08RGd6ymIBgIddkNA40H83o181mftKpZ7Gg2ArOgVXuJ16/YMZZPCi88LN0sXs1Idtzc2bhtnWUYjhlY7M08b4LXELEPRrPVTq6ylCqetlX4r7VLgTk5k1b+b6ztF3/BUdABTOW5fpH1U2qTQVO28p+A+Gy2hnkH+uB0NwxOCgYkf8UW/ljZBbr1E+QaB9ZUGsBHhO+iS0oljxIy92S1hJ; _cs_mk_ga=0.7063035713389325_1699579271515; utag_main__ga=4181960086.1699579272; utag_main_dc_visit=1; utag_main__ss=0%3Bexp-session; utag_main_dc_region=us-east-1%3Bexp-session; _abck=910BCDCEF48E18398EB2ACF3F09534FE~0~YAAQ0pfAFwUF0amLAQAAdNjWthVLfsDAIej8f4SWOXtfmRZRfBSPZXOPZsWHPlIRMH8VSBFIUvkcqU3VglZo4PxakkVPTZIzlYAehR/Zy2Yl90b3eDUsJvTcnM11JJeCwa4UTpi9UTiLdEs4Hx9RK0SQqQAkIZozYcV21SA6sZ/7vKpFCyNJND9GkOtxD7/j4wnNQHNkaHVOWCuMPA60r+2djY8clU+RrAnPGMiZk2TsMPU3FoREcxGGkUV1c3x0~1; utag_main__se=82%3Bexp-session; utag_main__st=1699581418519%3Bexp-session",
    #     "Dnt": "1",
    #     "Sec-Ch-Ua": '"Brave";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
    #     "Sec-Ch-Ua-Mobile": "?0",
    #     "Sec-Ch-Ua-Platform": '"macOS"',
    #     "Sec-Fetch-Dest": "document",
    #     "Sec-Fetch-Mode": "navigate",
    #     "Sec-Fetch-Site": "none",
    #     "Sec-Fetch-User": "?1",
    #     "Sec-Gpc": "1",
    #     "Upgrade-Insecure-Requests": "1",
    #     "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    # }
    # print(url)
    # try:
    #     req = requests.get(url, headers=headers, timeout=4)
    #     soup = BeautifulSoup(req.text, 'html.parser', headers=headers)
    #     print(soup)
    #     li_tags = soup.find_all("li", class_="grid-tile")
    #     print(li_tags)
    #     gatherRowData(li_tags)

    #     with open(asicsFile, "w") as jsonFile:
    #         json.dump(asicsPrices, jsonFile, indent=2)
    # except:
    #     print("Next")
    # Comment out the code for requesting the link
    # for i in range(0, 1400, 72):  # assuming there are 10 pages to scrape
    #     url = f"https://www.asics.com/us/en-us/mens-shoes/c/aa10200000/?start={i}&sz=72"
    #     collectData(url)

    # Run BeautifulSoup on example.html
    with open("example.html", "r") as file:
        soup = BeautifulSoup(file, 'html.parser')
        li_tags = soup.find_all("li", class_="grid-tile")
        gatherRowData(li_tags)

    with open(asicsFile, "w") as jsonFile:
        json.dump(asicsPrices, jsonFile, indent=2)

collectData("")