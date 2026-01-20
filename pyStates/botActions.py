import json
import os

class BotAction:
    def __init__(self, path, parameters=None):
        self.path = path
        self.name = os.path.basename(path)
        self.parameters = parameters or {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            # Expecting a list of dicts. If it's a dict, try to convert to list.
            if isinstance(loaded, dict):
                # common case: dict with numeric keys or single record
                if all(isinstance(v, dict) for v in loaded.values()):
                    self.data = list(loaded.values())
                else:
                    # fallback: wrap into single-element list
                    self.data = [loaded]
            elif isinstance(loaded, list):
                self.data = loaded
            else:
                self.data = []

            # prepare typed subsets
            self.auen = [e for e in self.data if e.get('Typ') == "Auen"]
            self.tiere = [e for e in self.data if e.get('Typ') == "Tier"]
            self.pflanzen = [e for e in self.data if e.get('Typ') == "Pflanze"]

            # collect keys/columns from first record
            first = self.data[0] if self.data else {}
            self.keys = [k for k in first.keys() if not (k.startswith("Name") or k in ['Typ','Gruppe'])]
            print(f"Loaded data for action '{self.name}' with {len(self.data)} entries.")
            print(f"Available keys: {self.keys}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file for action '{self.name}' not found.")
        except Exception as e:
            raise RuntimeError(f"Error loading data file '{path}': {e}")

    def extract_animal_or_plant(self, user_input):
        ents = self.find_entity(user_input, "Tier")
        if not ents:
            ents = self.find_entity(user_input, "Pflanze")
        return ents

    def tp_generell_extract_information(self, user_input):
        return self.find_entity(user_input)

    def find_entity(self, user_input, entity_type=None):
        try:
            terms = [user_input] + user_input.split(" ")
            for term in terms:
                term = term.strip()
                if not term:
                    continue
                print(f"Searching for term '{term}' and type '{entity_type}'")
                if entity_type is None:
                    ents = [e for e in self.data if term.lower() == (e.get('Name') or '').lower()]
                    if ents:
                        # determine type from first hit and restrict to that type
                        inferred = ents[0].get('Typ')
                        ents = [e for e in ents if e.get('Typ') == inferred]
                elif entity_type == "Tier":
                    ents = [e for e in self.tiere if term.lower() == (e.get('Name') or '').lower()]
                elif entity_type == "Pflanze":
                    ents = [e for e in self.pflanzen if term.lower() == (e.get('Name') or '').lower()]
                elif entity_type == "Auen":
                    ents = [e for e in self.auen if term.lower() == (e.get('Name') or '').lower()]
                else:
                    ents = [e for e in self.data if term.lower() == (e.get('Name') or '').lower() and e.get('Typ') == entity_type]

                if ents:
                    print(f"Found {len(ents)} entities for term '{term}' and type '{entity_type}'")
                    return ents
                else:
                    print("No matching entity found for:", term)
            return []
        except Exception as e:
            print(f"Error finding entity: {e}")
            return []

    def find_entity_key(self, user_input):
        try:
            terms = [user_input] + user_input.split(" ")
            for term in terms:
                term = term.strip()
                if not term:
                    continue
                keys = [k for k in self.keys if term.lower() in k.lower()]
                if keys:
                    return keys
            print("No matching keys found.")
            return []
        except Exception as e:
            print(f"Error finding keys: {e}")
            return []

    def get_entity_features(self,name,key):
        # check synonyms 
        #'Erkennungsmerkmale', 'Habitat', 'Fortpflanzung', 'Größe', 'Links', 'Familie', 
        # 'Gattung', 'Lebensraum', 'Klasse', 'Lebensweise', 'Nahrung', 'Feinde', 
        # 'Lebenserwartung', 'Schutz', 'Wissenswertes', 'Blütezeit', 'Verwendung',
        # 'Frucht', 'Vorkommen', 'Genießbarkeit', 'Ökologische Bedeutung', 'Giftigkeit', 
        # 'Alter', 'Gewicht', 'Überwinterung', 'Verhalten', 'Paarung']
        
        # key from intent:
            # tp_groesse, tp_habitat, tp_erkennungsmerkmale, tp_lateinischername, 
            # tp_lebenserwartung, tp_fortpflanzung, tp_aussehen
        searchImages = False

        if "habitat" in key.lower():
            searchKeys_ = ["Lebensraum", "Habitat"]
        elif "lebensraum" in key.lower():
            searchKeys_ = ["Lebensraum", "Habitat"]
        elif "merkmale" in key.lower():
            searchKeys_ = ["Erkennungsmerkmale", "Größe","Gewicht","Verhalten","Lebensweise"]
            searchImages = True
        elif "aussehen" in key.lower():
            searchKeys_ = ["Größe"]
            searchImages = True
        elif "paarung" in key.lower():
            searchKeys_ = ["Paarung", "Fortpflanzung"]
        elif "fortpflanzung" in key.lower():
            searchKeys_ = ["Paarung", "Fortpflanzung"]
        elif ("größe" in key.lower()) or ("groesse" in key.lower()):
            searchKeys_ = ["Größe"]
        elif "tp_lateinischername" in key.lower():
            searchKeys_ = ["Name_sci"]
        else:
            searchKeys_ = key.split("tp_")[-1]

        # make sure we have leading capital letters
        if isinstance(searchKeys_, str):
            searchKeys = [searchKeys_[0].upper() + searchKeys_[1:].lower()]
        else:
            searchKeys = [a[0].upper() + a[1:].lower() for a in searchKeys_]
            
        try:
            items = [e for e in self.data if name.lower() == (e.get('Name') or '').lower()]
            if items:
                values = {"text": [], "image": [],"audio": [], "video": [], "link": []}
                for key in searchKeys:
                    values["text"].extend([f.get(key) for f in items if key in f and f.get(key)])
                if searchImages:
                    # also collect image links if available
                    for f in items:
                        if 'Links' in f and f['Links']:
                            for l in f['Links']:
                                img = l.get("img",None)
                                if img:
                                    values["image"].append(img)
                                    break
                return values
            else:
                print("No matching entity found for features:", name)
                return []
        except Exception as e:
            print(f"Error getting features: {e}")
            return []   

if __name__ == "__main__":
    action = BotAction("../rawData/tiere_pflanzen_auen.json")
    for user_input in ["frosch habitat","fisch", "blume", "wasserfrosch", "auen","magerrasen"]:
        result = action.extract_animal_or_plant(user_input)
        if result:
            print("1:", [r.get('Name') for r in result])
            for f in ["Lebensraum","Merkmale","AUssehen"]:
                name = result[0].get('Name')
                features = action.get_entity_features(name,f)
                if features:
                    print(f"1: {name}   Feature '{f}':", features)
                    
        else:
            print("1: No results found.\n-----\n")

        result = action.find_entity(user_input, entity_type="Tier")
        if result:
            print("2:", [r.get('Name') for r in result])
        else:
            print("2: No results found.\n-----\n")

        result = action.tp_generell_extract_information(user_input)
        if result:
            print("3: Type detected:", result[0].get('Typ'))
            print("3:", [r.get('Name') for r in result])
        else:
            print("3: No results found.\n-----\n")
        print("-----\n")

        result = action.find_entity_key(user_input)
        if result:
            print("4:", result)
        else:
            print("4: No results found.\n-----\n")
