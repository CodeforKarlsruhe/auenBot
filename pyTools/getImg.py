# load images from wikipedia
import requests
import json
import time
import os

with open("../rawData/aniPlants.json") as f:
    aniplants = json.load(f)
    
for item_ in aniplants:
    item = aniplants[item_]
    name = item["name"]
    links = item.get("links", [])
    for l in links:
        if "upload.wikimedia.org" in l:
            extension = l.split(".")[-1]
            filename = f"{name}.{extension}"
            if os.path.exists(f"../media/img/{filename}"):
                print(f"File {filename} already exists, skipping...")
                continue
            print(f"Downloading {filename} from {l}")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(l,headers=headers)
            if response.status_code == 200:
                with open(f"../media/img/{filename}", "wb") as img_file:
                    img_file.write(response.content)
            else:
                print(f"Failed to download {l}: Status code {response.status_code}")
            time.sleep(15)  # be polite to the server