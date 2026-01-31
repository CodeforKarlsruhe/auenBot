import json
import pandas as pd

source_dir = "../rawData/"
dest_dir = "./data/"

df = pd.read_json(source_dir + "intents.json")

# clean intent names
df.intent = df.intent.str.strip().str.lower()

df["handler"] = "default"
df["action"] = None
df["provides"] = None
df["requires"] = None


def getElem(x,key="feature",tp = None):
    elem = x.split("_")[-1].strip()
    item = {key: elem}
    if tp is not None:
        item["type"] = tp
    return item


# for these intents, the options/feature list is too large to display (maybe not for aue)
df.loc[df.intent.str.startswith("tp_"),"requires"] = "entity,feature"
mask = df.intent.str.startswith("tp_")
df.loc[mask, "provides"] = df.loc[mask, "intent"].apply(lambda s: getElem(s,"feature","TP"))
df.loc[df.intent.str.startswith("tp_"),"handler"] = "complete"
df.loc[df.intent.str.startswith("tp_"),"action"] = "bio_feature"

df.loc[df.intent.str.startswith("tiere_"),"requires"] = "entity,feature"
mask = df.intent.str.startswith("tiere_")
df.loc[mask, "provides"] = df.loc[mask, "intent"].apply(lambda s: getElem(s,"feature","Tiere"))
df.loc[df.intent.str.startswith("tiere_"),"handler"] = "complete"
df.loc[df.intent.str.startswith("tiere_"),"action"] = "bio_feature"
df.loc[df.intent.str.startswith("pflanzen_"),"requires"] = "entity,feature"
mask = df.intent.str.startswith("pflanzen_")
df.loc[mask, "provides"] = df.loc[mask, "intent"].apply(lambda s: getElem(s,"feature","Pflanzen"))
df.loc[df.intent.str.startswith("pflanzen_"),"handler"] = "complete"
df.loc[df.intent.str.startswith("pflanzen_"),"action"] = "bio_feature"
df.loc[df.intent.str.startswith("aue_"),"requires"] = "entity"
mask = df.intent.str.startswith("aue_")
df.loc[mask, "provides"] = df.loc[mask, "intent"].apply(lambda s: getElem(s,"entity","Aue"))
df.loc[df.intent.str.startswith("aue_"),"action"] = "bio_feature"


# some intents require a feature and have options for features, and/or actions
df.loc[df.intent == "messdaten","requires"] = "type"
df.loc[df.intent == "messdaten","options"] = "messdaten" # => load from options_messdaten.json
df.loc[df.intent == "messdaten","handler"] = "complete"
df.loc[df.intent == "messdaten","action"] = "messdaten"

df.loc[df.intent == "anreise","requires"] = "type"
df.loc[df.intent == "anreise","options"] = "anreise" # => load from options_anreise.json
df.loc[df.intent == "anreise","handler"] = "complete"
df.loc[df.intent == "anreise","action"] = "anreise"

df.loc[df.intent == "expertise","requires"] = "target"
df.loc[df.intent == "expertise","options"] = "expertise" 
df.loc[df.intent == "expertise","handler"] = "complete"  # => decodes and routes to following intent


# kinder_ has feature but output differs only by link. so far
# df.loc[df.intent.str.startswith("kinder_"),"requires"] = "feature"
mask = df.intent.str.startswith("kinder_")
df.loc[mask, "provides"] = df.loc[mask, "intent"].apply(lambda s: getElem(s,"feature"))

df.loc[df.intent == "kinder","action"] = "kinder"
df.loc[df.intent == "ausstellungen","action"] = "ausstellung"

# forward messdaten_welche into messdaten via intent
df.loc[df.intent == "messdaten_welche","handler"] = "complete"
df.loc[df.intent == "messdaten_welche","requires"] = "target"
df.loc[df.intent == "messdaten_welche","provides"] = df.loc[df.intent == "messdaten_welche","provides"].apply(lambda x: {"target":"messdaten"})



df.to_json(dest_dir + "intents.json",orient="records",indent=2,force_ascii=False)

# write action list
with open(dest_dir + "actions.json","w",encoding="utf-8") as f:
    json.dump([a for a in list(df.action.unique()) if a != None],f, ensure_ascii=False, indent=2)

dft = df[["id","intent","text"]]
dft.to_json(dest_dir + "intents_text.json",orient="records",indent=2,force_ascii=False)
dfi = df.drop(columns="text")

dfi.to_json(dest_dir + "intents_raw.json",orient="records",indent=2,force_ascii=False)
def strip_type(x):
    return x.split("_")[-1].strip()

x = dfi[df.intent.str.startswith("tiere_")].copy()
x.intent = x.intent.apply(strip_type)
x.intent.to_json(dest_dir + "feature_tiere.json",index=False,orient="records",indent=2,force_ascii=False)

x = dfi[df.intent.str.startswith("pflanzen_")].copy()
x.intent = x.intent.apply(strip_type)
x.intent.to_json(dest_dir + "feature_pflanzen.json",index=False,orient="records",indent=2,force_ascii=False)
x = dfi[df.intent.str.startswith("aue_")].copy()
x.intent = x.intent.apply(strip_type)
x.intent.to_json(dest_dir + "feature_aue.json",index=False,orient="records",indent=2,force_ascii=False)

x = dfi[df.intent.str.startswith("tp_")].copy()
x.intent = x.intent.apply(strip_type)
x.intent.to_json(dest_dir + "feature_tp.json",index=False,orient="records",indent=2,force_ascii=False)
#####
x = dfi[df.intent.str.startswith("kinder_")].copy()
x.intent = x.intent.apply(strip_type)
x.intent.to_json(dest_dir + "feature_kinder.json",index=False,orient="records",indent=2,force_ascii=False)
#####
tf = pd.read_json(source_dir + "tiere_pflanzen_auen.json")

tt = tf[tf.Typ == "Tier"]
tp = tf[tf.Typ == "Pflanze"]
ta = tf[tf.Typ == "Auen"]



tt.Name.to_json(dest_dir + "entity_tiere.json",orient="records",indent=2,force_ascii=False)
tp.Name.to_json(dest_dir + "entity_pflanzen.json",orient="records",indent=2,force_ascii=False)
ta.Name.to_json(dest_dir + "entity_aue.json",orient="records",indent=2,force_ascii=False)

tp_entities = list(tt.Name.unique()) + list(tp.Name.unique())
with open(dest_dir + "entity_tp.json","w",encoding="utf-8") as f:
    json.dump(tp_entities,f,ensure_ascii=False, indent=2)

k = list(tt.Klasse.unique())
with open(dest_dir + "klasse_tiere.json","w",encoding="utf-8") as f:
    json.dump(k,f,ensure_ascii=False, indent=2)
k = [x for x in tt.Unterklasse.unique() if pd.notna(x)]
with open(dest_dir + "unterklasse_tiere.json","w",encoding="utf-8") as f:
    json.dump(k,f,ensure_ascii=False, indent=2)
k = list(tp.Klasse.unique())
with open(dest_dir + "klasse_pflanzen.json","w",encoding="utf-8") as f:
    json.dump(k,f,ensure_ascii=False, indent=2)
k = [x for x in tp.Unterklasse.unique() if pd.notna(x)]
with open(dest_dir + "unterklasse_pflanzen.json","w",encoding="utf-8") as f:
    json.dump(k,f,ensure_ascii=False, indent=2)

