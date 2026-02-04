import requests
import json 
import os
import time


        

def process_all():
    with open("../rawData/tiere_pflanzen_auen.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Processing {len(data)} entries")

    for entry in data:
        name = entry.get("Name", "Unknown")
        print(f"Processing entry: {name}")
        links = entry.get("Links", [])
        for link in links:
            img_url = link.get("img")
            if img_url and "wikimedia.org" in img_url:
                try:
                    print(f"  Downloading image from {img_url}")
                    headers = {"User-Agent": "auenBot/1.0"}
                    response = requests.get(img_url,headers=headers)
                    print(f"    HTTP Status Code: {response.status_code}")
                    response.raise_for_status()  # Raise an error for bad responses

                    img_data = response.content
                    img_name = os.path.basename(img_url.split("?")[0])  # Get the image file name
                    local_path = os.path.join("./wikiImgs/", img_name)

                    with open(local_path, "wb") as img_file:
                        img_file.write(img_data)

                    link["img_local"] = local_path  # Update the entry with the local image path
                    print(f"    Saved to {local_path}")
                    time.sleep(1)   

                except Exception as e:
                    print(f"    Failed to download {img_url}: {e}")
                    

                time.sleep(1)  # Be polite and avoid overwhelming the server

    with open("../rawData/tiere_pflanzen_auen_local.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)    


if __name__ == "__main__":
    process_all()



