import json 
import os
from datetime import datetime, timezone
import re
import shutil


src = "./wikiImgs/"
dst = "./wikiImgs2/"

os.makedirs(dst, exist_ok=True)
        


def process_all():
    with open("../rawData/tiere_pflanzen_auen.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Processing {len(data)} entries")

    errs = []

    for entry in data:
        name = entry.get("Name", "Unknown")
        print(f"Processing entry: {name}")
        links = entry.get("Links", [])
        for link in links:
            img_url = link.get("img")
            if img_url and "wikimedia.org" in img_url:
                try:
                    segments = [s for s in img_url.split("/") if s]
                    imgName = segments[-1] if segments else None
                    
                    # imgName = getPageInfo(img_url)
                    if not imgName:
                        print(f"    Could not extract image name from URL: {img_url}")
                        continue

                    # try a few name variants to find the file in src (case-insensitive)
                    candidates = {imgName, imgName.replace(' ', '_'), imgName.replace('_', ' ')}
                    candidates_lower = {c.lower() for c in candidates if c}

                    source_path = None
                    found_fname = None
                    for root, _, files in os.walk(src):
                        for f in files:
                            if f.lower() in candidates_lower:
                                source_path = os.path.join(root, f)
                                found_fname = f
                                break
                        if source_path:
                            break

                    if not source_path:
                        print(f"    Source file for {imgName} not found in {src}")
                        errs.append({"name":entry["Name"], "imgName": imgName})
                        continue

                    # sanitize entry name and build destination filename
                    raw_name = entry.get("Name", "Unknown")
                    safe_name = re.sub(r'[\\/*?:"<>|]', "", raw_name).strip() or "Unknown"
                    safe_name = re.sub(r'\s+', '_', safe_name)  # replace spaces with underscores
                    ext = os.path.splitext(found_fname)[1] or os.path.splitext(imgName)[1] or ""
                    dest_filename = f"{safe_name}{ext}"
                    dest_path = os.path.join(dst, dest_filename)

                    try:
                        shutil.copy2(source_path, dest_path)
                        print(f"    Copied {source_path} -> {dest_path}")
                    except Exception as e:
                        print(f"    Failed to copy local file: {e}")
                        continue

                    # update imgName so subsequent local_path uses the renamed file
                    localImgName = dest_filename
                    local_path = os.path.join(dst, localImgName)
                    print(f"  Copying image from {img_url}")

                    link["img_local"] = localImgName  # Update the entry with the local image path
                    print(f"    Saved to {local_path}")

                except Exception as e:
                    print(f"    Failed to copy local file for {img_url}: {e}")
                    
    print(f"Finished processing with {len(errs)} errors.")
    if errs:
        print("Errors for the following image names:")
        for err in errs:
            print(f" - {err}")

    with open(os.path.join(dst, "tiere_pflanzen_auen_localimg.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)    


if __name__ == "__main__":
    process_all()



