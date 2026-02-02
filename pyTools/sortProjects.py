import json
from collections import defaultdict
from pathlib import Path

infile = Path("./Projektdateien KarlA 12.11.25/project-database-version-dump.json")
outfile = Path("./project-grouped.json")

with infile.open("r", encoding="utf-8") as f:
    data = json.load(f)

items = data.get("documents", {}).get("IntentItem", [])
groups = defaultdict(list)

for it in items:
    intent = it.get("intentId")
    text = it.get("text")
    if not intent or not text:
        continue
    if text not in groups[intent]:
        groups[intent].append(text)
    entitities = it.get("entities", [])
    if len(entitities) == 0:
        continue
    else:
        etext = entitities[0].get("text")
        if etext and etext not in groups[intent]:
            groups[intent].append(f"Entity - {etext}")

with outfile.open("w", encoding="utf-8") as f:
    json.dump(dict(groups), f, ensure_ascii=False, indent=2)

print(f"Wrote {len(groups)} intents to {outfile}")

