import json
import pandas as pd

with open("tiere_pflanzen_auen.json") as f:
    tpa = json.load(f)
    
t = pd.read_csv("tiere.csv",sep=";")
p = pd.read_csv("pflanzen.csv",sep=";")

def lmerge(x):
    for l in ["Lebensraum2","Lebensraum3","Lebensraum4"]:
        if isinstance(x[l],str):
            x["Lebensraum"] += "." + x[l]
    return x

print("Tiere")
t = t.apply(lmerge,axis=1)
t = t.drop(columns=["Lebensraum2","Lebensraum3","Lebensraum4"])
print("Pflanzen")
p = p.apply(lmerge,axis=1)
p = p.drop(columns=["Lebensraum2","Lebensraum3","Lebensraum4"])

t.to_json("tiere.json",indent=2,orient="records")
p.to_json("pflanzen.json",indent=2,orient="records")


def findItem(x,y):
    if x is None:
        return pd.DataFrame()
    x_lower = x.lower()
    if y is not None:
        y_lower = y.lower()
    item = t[(t.Name.str.lower() == x_lower) | (t.Name_alt.str.lower() == x_lower)]
    if item.empty and not y is None:
        item = t[(t.Name.str.lower() == y_lower) | (t.Name_alt.str.lower() == y_lower)]
    if item.empty:
        item = p[(p.Name.str.lower() == x_lower) | (p.Name_alt.str.lower() == x_lower)]
    if item.empty and not y is None:            
            item = p[(p.Name.str.lower() == y_lower) | (p.Name_alt.str.lower() == y_lower)]

    # add gemeiner 
    if item.empty:
        item = t[(t.Name.str.lower() == "gemeiner " + x_lower) | (t.Name_alt.str.lower() == "gemeiner " + x_lower)]
    if item.empty:
        item = p[(p.Name.str.lower() == "gemeiner " + x_lower) | (p.Name_alt.str.lower() == "gemeiner " + x_lower)]

    if item.empty:
        print("NOT FOUND:", x)
    return item

for item in tpa:
    if item["Typ"] != "Auen":
        tp = findItem(item["Name"],item.get("Name_alt",None))
