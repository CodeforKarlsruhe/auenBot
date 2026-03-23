import json
import os
with open("rawData/tiere_pflanzen_auen.json") as f:
    tpa = json.load(f)
    

os.makedirs("./tpa_docs",exist_ok=True)    
    
for i in tpa:
    name = i["Name"].strip().replace(" ","_").lower()
    keys = list(i.keys())
    tx = f'# {i["Name"]}\n## Lateinischer Name: {i["Name_sci"]}\n'
    for k in list(keys):
        if k.lower().startswith("name"):
            continue
        if isinstance(i[k],str):
            content = i[k].replace("["," ").replace("]"," ") 
            tx += f"{k}:{content}.\n"

    with open(f"tpa_docs/{name}.md","w") as f:
        f.write(tx)
        
