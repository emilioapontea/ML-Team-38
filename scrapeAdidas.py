import requests
import json
import os


headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.6",
    "Cache-Control": "max-age=0",
    "Cookie": "adidas_country=us; geo_country=US; onesite_country=US; gl-feat-enable=CHECKOUT_PAGES_ENABLED; badab=false; akacd_phased_PDP=3876180761~rv=87~id=e4e76d0ebc1ddc611beadbb96b36e56d; notice_preferences=2; akacd_Phased_www_adidas_com_Generic=3876180761~rv=57~id=f8b15347c22e4601bbc5dcd62e3350f1; x-original-host=adidas.co.uk; x-site-locale=en_GB; mt.v=2.888724654.1698727964142; wishlist=%5B%5D; ab_inp=a; AMCVS_7ADA401053CCF9130A490D4C%40AdobeOrg=1; s_cc=true; akacd_phased_PLP=3876180815~rv=66~id=3fca480fed840154b8647bb9b9bed006; x-commerce-next-id=ec190d2b-2367-4654-9d83-526caa6e5981; pagecontext_cookies=; pagecontext_secure_cookies=; persistentBasketCount=0; userBasketCount=0; __olapicU=3ea2f38f0d073a9599a05ddb2d904f95; notice_preferences=2; ab_qm=b; RES_TRACKINGID=54152921521769399; ResonanceSegment=1; newsletterShownOnVisit=true; geo_ip=128.61.160.41; AKA_A2=A; checkedIfOnlineRecentlyViewed=true; geo_state=GA; geo_coordinates=lat=33.7486, long=-84.3884; s_sess=%5B%5BB%5D%5D; AWSELB=95ADB7E50C84216D4F0382FB851EE9236F353F155AB99C37D811666742F6B0D5317229A909719215C06F654D9EB4628156B4A3BE7E2267820134FC12BCE4DC8A75B854AC43; AWSELBCORS=95ADB7E50C84216D4F0382FB851EE9236F353F155AB99C37D811666742F6B0D5317229A909719215C06F654D9EB4628156B4A3BE7E2267820134FC12BCE4DC8A75B854AC43; bm_sz=E9E9F359840F024D86AC478DA64219D2~YAAQHiXRF2Im862LAQAAPpsathVVfpasx1bf6s3IYlO0qA9hOqK/UYbZHdqYZkgzqDyyz8I6EwzyA0SoyfK3D2IeHy+4SlAft06LTAa0HMa1IVjPIKR5Bj/UsB+rT3zYUnhDBXPpZy8ZxqKZWCPSQil5YbIvVfXtkBVRT7ZqEHGW07fnm6bpNWEKste+p/WZR5SiZFM9T1KNCwqBfk1qG2wQ3eLbTI4Wqf2Oxk6E177Ak7Zi+/5NdcdZWqey/h40TA4OsWwJOYBZjIj/y3I4iwpwsPHxYkQqVe6Mj9NefdpFRUCphD+GUPE1KohSijk0enj9wantfZ6Gmmy9GY17DRFMCvlxlbnOePm/Yv/24DJOJfzTWYmdGo1wdBFqOHlAQ4/UEMkeRWzTxaFhMJu64cra8e5Irolq4Nmmdc082itdQjPlHS6clOtxr/3C1AsYFY6boC+KpeiFzQEHjzXRIwYzpr6nCWnDZMy/8lvHA8uhjv3eAQ+Qoqt03//qm25p4+snX1HPMx1lNAJoco73IjnsbYFpEVprJz4IusAfTN7RcNo4V/07uA==~4534342~3425074; _abck=2932DED23C2DE6589F7E176ACD9807C7~-1~YAAQHiXRF/om862LAQAAE6oatgq9EzWiznvtdSyBILsjqY2tke3lMT/YvSC96pGu1ZEp/+WH5P7DslxFZNzWXHjELOePvLkCs92m4Ltxru9lgiRbJu8kUE1wK4tjuDGy+4kqH7yVPWAkj5nMzL0Utk2SOJQ7QhpLLLpC7eKX5hLPeIdbZpzwcEwqC0p+NN5y/KtAE/Ef2i/NE94rM1BuAdJlJBWKAh+6y4765FKIr60nXSuJfoCd05OA+TZnLsGVWoiNjghY38eB22V4QYnJYva7vfH54hro3TRQhk5K93R5ZXhLb03WLSUCNRCC5FTTuhOtygkhvtXTkziQbL4Mbp/5vQswijUW+bFvB24BIInLu47ktKvt9fR/IrDEazgedwyCExqggkviCeQb15uFRWwXaDHUjNAVu8H3fSwDLnIi9nkJs3acYoL5V+j2Q1iL36rzXErd2DPilw==~-1~-1~1699570821; UserSignUpAndSave=18; bm_sv=8CF291D4CF2780B4AAE4A11B6BB53015~YAAQVSABF/XrqqmLAQAAkq0athWXJXyEaUTj5HUOUVUehurPkoJNij9ULQ3RwIq2j+QTB9Bqoor52WPIxdkOL7l/04PAD4wvd9osXVibWuZOEx9OzFvmvVrcGvfAUW8Ak66xjhS5ai5sErcgHz2ZG4Ww/KR2LpyOp3E9xcxC5+dGm+jW6ma3VWvxBA5ZO3h0qNdJTO6yxTHWtA0gXDvVdo0kzQynjSwXt9RJCiZZvZwxmrVcxKTQGJYyEjCidSse~1; ak_bmsc=C498CAD9714D23ED2A22920BA3DC89C6~000000000000000000000000000000~YAAQHiXRFxon862LAQAAA64athXUCOma2NU4oTEvUgOG7SlI9or37E1i6P73xN4OeTfwicOPLGcsL/rTC9UhsX463vwPAMiV9kViHdxefZyBdndLPHB8/KxYtiM6WAKJ+HMbSecKnJJ21vTHSxOWLElxDzUY2j6U1w73kQUgaVhTkdQeQa9PuhLdMLrgUQypA4JnVuc6IBGw5V35/qWwE1f7MX9LVgtulCQphy/B6nWSk8FGmZr4HQOUDTfEKhmEZPLsJ55dbOqSfgT5Ql7AB776TDltx0W2uRoLa005ixlEK7cZ/Su6hn2u1Av9y51JvaTNfYCa6EFyHLRq+5w7WuUYxPP7QAXvUME6svWwUt37ye0k3EEGu/Cf92Wlvl2mcFUjZ8SpgW0=; s_pers=%20s_vnum%3D1701406800467%2526vn%253D3%7C1701406800467%3B%20pn%3D2%7C1702158671467%3B%20s_invisit%3Dtrue%7C1699569086195%3B; AMCV_7ADA401053CCF9130A490D4C%40AdobeOrg=-227196251%7CMCIDTS%7C19671%7CMCMID%7C83234778933402507797460356648177499547%7CMCAID%7CNONE%7CMCOPTOUT-1699574486s%7CNONE; utag_main=v_id:018babefdf2c001282ba101eefbc04075001906d00b54$_sn:3$_se:10%3Bexp-session$_ss:0%3Bexp-session$_st:1699569085482%3Bexp-session$ses_id:1699564899965%3Bexp-session$_pn:5%3Bexp-session$ab_dc:TEST%3Bexp-1704751285674$_vpn:7%3Bexp-session$ttdsyncran:1%3Bexp-session$dcsyncran:1%3Bexp-session$dc_visit:1$dc_event:19%3Bexp-session$_prevpage:PLP%7CG_MEN%7CPR_SHOES%3Bexp-1699570886150",
    "Dnt": "1",
    "Sec-Ch-Ua": '"Brave";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Sec-Gpc": "1",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}


menURL = "https://www.adidas.com/api/plp/content-engine?sitePath=us&query=men-shoes&start=0"
# * https://www.adidas.com/api/plp/content-engine?sitePath=us&query=men-athletic_sneakers&start=720
adidasPrices = {}

adidasFile = "./scrapedData/adidasPrices.json"

if os.path.exists(adidasFile) and os.path.getsize(adidasFile) > 0:
    with open(adidasFile, "r") as jsonFile:
        adidasPrices = json.load(jsonFile)
else:
    print("File is empty or does not exist")

print(adidasPrices)

def gatherRowData(jsonData, prevData=adidasPrices):
    for each in jsonData:
        print(each)
        price = str(each.get("price"))
        portraitURL = (each.get("image").get("src")).replace("w_280,h_280,f_auto,q_auto:sensitive/", "")
        # print(type(price))
        if price not in prevData.keys():
            prevData[price] = [portraitURL]
        elif portraitURL not in prevData[price]:
            prevData[price].append(portraitURL)  # Append to the existi


import os
import json

adidasFiles = os.listdir("./adidasStuff")

for file in adidasFiles:
    with open(f"./adidasStuff/{file}", "r") as jsonFile:
        adidasReq = json.load(jsonFile)
        jsonData = adidasReq.get("raw").get("itemList").get("items")
        gatherRowData(jsonData=jsonData)
    with open(adidasFile, "w") as jsonFile:
        json.dump(adidasPrices, jsonFile, indent=2)

with open(adidasFile, "w") as jsonFile:
    json.dump(adidasPrices, jsonFile, indent=2)