    
import json
from urllib import response
import requests
import private as pr 
import base64
import re
from pathlib import Path

API_URL = "https://api.deepinfra.com/v1/openai/images/generations"    
API_KEY = pr.apiKey
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}   
def generate_image(prompt, size="512x512", model="black-forest-labs/FLUX-1-schnell", n=1):
    payload = {
        "prompt": prompt,
        "size": size,
        "model": model,
        "n": n
    }
    response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")  
    

with open("../rawData/tiere_pflanzen_auen.json", "r") as f:
    data = json.load(f)

img_dir = Path("fantasyImages")
img_dir.mkdir(parents=True, exist_ok=True)
    
for item in data:
    name = item["Name"]
    name_eng = item.get("Name_Eng", name)
    classType = item.get("Klasse", "")
    if item["Typ"] == "Tier":
        if "Erkennungsmerkmale" in item:
            description = item.get("Erkennungsmerkmale", "")
        elif "Habitat" in item:
            description = item.get("Habitat", "")
        elif "Lebensweise" in item:
            description = item.get("Lebensweise", "")  
        else:
            description = ""
        # bats and grasshopper need special prompts with flux-1
        if "fledermaus" in name.lower():
            description = item.get("Erkennungsmerkmale", "")
            prompt = f"Darstellung einer Fledermaus (bat, {name_eng}), Foto in natürlicher Umgebung. {description}."
        elif "schrecke" in name.lower():
            description = item.get("Erkennungsmerkmale", "")
            prompt = f"Darstellung eines Grashüpfers (grasshopper, {name_eng}) , Foto in natürlicher Umgebung. {description}."
        elif "muschel" in name.lower():
            description = item.get("Erkennungsmerkmale", "")
            prompt = f"Darstellung einer Muschel (mussel, {name_eng}) , Foto in natürlicher Umgebung. {description}."
        elif "schnake" in name.lower():
            description = item.get("Erkennungsmerkmale", "")
            prompt = f"Darstellung einer Schnake (cranefly, {name_eng}) , Foto in natürlicher Umgebung. {description}."
        else:            
            prompt = f"Tier der Klasse {classType}, Spezies {name} ({name_eng}) aus der Familie {item.get('Familie', '')}, Foto in natürlicher Umgebung. {description}."
    elif item["Typ"] == "Pflanze":
        if "Erkennungsmerkmale" in item:
            description = item.get("Erkennungsmerkmale", "")
        elif "Vorkommen" in item:
            description = item.get("Vorkommen", "")
        elif "Wissenswertes" in item:
            description = item.get("Wissenswertes", "")  
        else:
            description = ""
        prompt = f"Pflanze der Klasse {classType}, Spezies {name}, Foto in natürlicher Umgebung. {description}."
    else:
        if "Erkennungsmerkmale" in item:
            description = item["Erkennungsmerkmale"]
        else:
            description = ""
        prompt = f"Biotop am Rhein, genannt {name}, Naturaufnahme. {description}."
        

    try:
        result = generate_image(prompt, size="512x512", model="black-forest-labs/FLUX-1-schnell", n=1)
        b64_image = result["data"][0]["b64_json"]
        safe_name = re.sub(r'[^A-Za-z0-9._ÄÖÜäöüß-]+', '_', name).strip('_') or "image"
        img_path = img_dir / f"{safe_name}.jpg"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(b64_image))
        item["generated_image"] = b64_image
        print(f"Generated image for prompt: {prompt}")
    except Exception as e:
        print(f"Failed to generate image for prompt: {prompt}. Error: {e}")
        pass

