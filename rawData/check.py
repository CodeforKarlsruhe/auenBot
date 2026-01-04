import os,json
import pandas as pd


with open("tiere_pflanzen_auen.json") as f:
    tpa = json.load(f)

with open("aniPlants.json") as f:
    ap = json.load(f)

def strip_all(obj):
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, dict):
        return {k: strip_all(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_all(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(strip_all(v) for v in obj)
    return obj

#tpa = [strip_all(item) for item in tpa]
#ap = [strip_all(item) for item in ap]

#with open("tiere_pflanzen_auen.json", "w", encoding="utf-8") as f:
#    json.dump(tpa, f, ensure_ascii=False, indent=2)

#with open("aniPlants.json", "w", encoding="utf-8") as f:
#    json.dump(ap, f, ensure_ascii=False, indent=2)


tpaSet = set(a["Name"].lower() for a in tpa)
apSet = set(a["Name"].lower() for a in ap)

#check missing in tpa
missingA = [a for a in tpaSet if not a in apSet]
print("Missing in ap:", missingA)
    
#check missing in ap
missingB = [a for a in apSet if not a in tpaSet]
print("Missing in tpa:", missingB)
