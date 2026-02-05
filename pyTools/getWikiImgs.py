import requests
import json 
import os
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone


        

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
                #if not "220px-" in img_url:
                #        print(f"  Skipping image {img_url} because it does not contain '220px-' (not a thumbnail)")
                #        continue
                try:
                    print(f"  Downloading image from {img_url}")
                    headers = {"User-Agent": "auenBot/1.0"}
                    response = requests.get(img_url,headers=headers)
                    print(f"    HTTP Status Code: {response.status_code}")
                    # Handle 429 Too Many Requests by honoring Retry-After and retrying a few times
                    max_retries = 5
                    for attempt in range(max_retries):
                        if response.status_code != 429:
                            break

                        retry_after = response.headers.get("Retry-After")
                        delay = 60  # default fallback
                        if retry_after:
                            retry_after = retry_after.strip()
                            if retry_after.isdigit():
                                delay = int(retry_after)
                            else:
                                try:
                                    dt = parsedate_to_datetime(retry_after)
                                    if dt.tzinfo is None:
                                        dt = dt.replace(tzinfo=timezone.utc)
                                    now = datetime.now(timezone.utc)
                                    secs = (dt - now).total_seconds()
                                    if secs > 0:
                                        delay = int(secs)
                                except Exception:
                                    # If parsing fails, fall back to default delay
                                    pass

                        print(f"    Received 429, sleeping for {delay} seconds before retrying (attempt {attempt+1}/{max_retries})")
                        time.sleep(delay)
                        response = requests.get(img_url, headers=headers)

                    if response.status_code == 429:
                        raise Exception("Received 429 Too Many Requests after retries")
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



