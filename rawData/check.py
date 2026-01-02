import os,json
import pandas as pd

with open("tiereKeys.json") as f:
    tKeys = set(json.load(f))
with open("pflanzenKeys.json") as f:
    pKeys = set(json.load(f))

props = tKeys
props.update(pKeys)

props = sorted([x.lower() for x in list(props)])

with open("tiere_pflanzen_auen.json") as f:
    daten = json.load(f)

with open("taskList.json") as f:
    tasks = json.load(f)

def checkKey(item):
    itemKeys = item.keys()
    missing 
