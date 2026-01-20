import json
import os
import random

# original actions expect these helper functions to exist
def old_generate_answer_animal_or_plant(res, intent_name):
    """
    Generate answer text based on entity data and intent name.
    This is a placeholder implementation; replace with actual logic.
    """
    if not res:
        return {"text": "Keine Informationen gefunden.", "has_shout": False}

    name = res[0].get('Name') if isinstance(res, list) and res else str(res)
    text = f"Informationen zu {name} für die Anfrage '{intent_name}'."
    has_shout = 'Rufe' in intent_name  # Example condition
    image = None
    if intent_name == "TP_Aussehen":
        image = res[0].get('Links', [{}])[0].get('img', None) if isinstance(res, list) and res else None

    return {
        "text": text,
        "has_shout": has_shout,
        "image": image,
        "last_intent": intent_name
    }
    
def old_action_TP_generateAnswer(api, tracker, bot_action):
    """
    Generate answer for a specific animal/plant based on tracker message.
    Expects helper function `generate_answer_animal_or_plant(res, intent_name)` to exist.
    """
    res = bot_action.extract_animal_or_plant(tracker.latest_message["text"])
    if res:
        # res may be a list of entity dicts — use first entry's name for buttons/text
        name = res[0].get('Name') if isinstance(res, list) and res else str(res)
        response = old_generate_answer_animal_or_plant(res, tracker.latest_message["intent"]["name"])
        buttons = []
        if response.get("last_intent") != "TP_Definition":
            buttons.append(api.create_button("Info " + name, "Was ist ein " + name))
        if response.get("last_intent") != "TP_Aussehen":
            buttons.append(api.create_button("Bild " + name, "Zeige mir ein Foto von " + name))
        if response.get("last_intent") != "TP_LateinischerName":
            buttons.append(api.create_button("Lateinischer Name " + name, "Wie ist der lateinische Name von " + name))
        if response.get("last_intent") != "Tiere_Rufe" and response.get("has_shout"):
            buttons.append(api.create_button("Tierruf " + name, "Welche Geräusche macht " + name))

        if response.get("last_intent") == "TP_Aussehen":
            if response.get("image"):
                api.bot_utter(response.get("text"), image=response.get("image"), buttons=buttons)
            else:
                api.bot_utter("Tut mir leid, von " + name + " habe ich leider keine Bilder.", buttons=buttons)
        else:
            api.bot_utter(response.get("text"), buttons=buttons)
    else:
        api.bot_utter("Ich habe leider nicht verstanden, welches Tier/welche Pflanze du meinst.")


def old_tp_generell_generate_answer(entities):
    """
    Generate a list of matching animal/plant names based on entities.
    entities is expected to be like: [category_name, lebensraum_name]
    This function will safely check for global data structures (ANIMAL, PLANT,
    lr_categories, animal_categories) and return an empty list if they are not present.
    """
    result = []
    if not entities or not entities[0]:
        return result

    category = entities[0]
    lr = entities[1] if len(entities) > 1 else None

    ANIMAL = globals().get('ANIMAL')
    PLANT = globals().get('PLANT')
    lr_categories = globals().get('lr_categories')
    animal_categories = globals().get('animal_categories')

    # helper to check if lr matches an entry's lr index safely
    def lr_matches(entry_lr_list):
        if not lr:
            return True
        if not lr_categories or lr not in lr_categories:
            return False
        try:
            idx = lr_categories.index(lr)
        except ValueError:
            return False
        return idx in entry_lr_list

    try:
        if category == "Tiere":
            if not ANIMAL:
                return result
            for name, meta in ANIMAL.items():
                try:
                    if lr_matches(meta[4]):
                        result.append(name)
                except Exception:
                    continue
        elif category == "Pflanzen":
            if not PLANT:
                return result
            for name, meta in PLANT.items():
                try:
                    if lr_matches(meta[4]):
                        result.append(name)
                except Exception:
                    continue
        else:
            # treat category as an animal subcategory name if animal_categories present
            if ANIMAL and animal_categories and category in animal_categories:
                cat_index = animal_categories.index(category)
                for name, meta in ANIMAL.items():
                    try:
                        if str(meta[3]) == str(cat_index) and lr_matches(meta[4]):
                            result.append(name)
                    except Exception:
                        continue
    except Exception:
        # on any error, return what we have or empty list
        return result

    return result


def old_tp_generell_extract_information(latest_msg):
    """
    Extract a (category, lebensraum) tuple from latest_msg.
    This is a simple, deterministic fallback replacement for fuzzy matching.
    It looks for known animal subcategories first, then general categories,
    and then for lr_categories entries.
    Returns a list like [category_or_None, lr_or_None].
    """
    result = [None, None]
    text = (latest_msg or "").lower()

    animal_cats = globals().get('animal_categories') or []
    lr_cats = globals().get('lr_categories') or []

    # try to match an animal subcategory (exact substring match)
    for cat in animal_cats:
        if cat and cat.lower() in text:
            result[0] = cat
            break

    # if no subcategory found, try general categories
    if not result[0]:
        for cat in ["Tiere", "Pflanzen", "Bäume", "Blumen"]:
            if cat.lower() in text:
                # normalize to either "Tiere" or "Pflanzen"
                result[0] = "Tiere" if cat == "Tiere" else "Pflanzen"
                break

    # try to find a lebensraum (lr) entry
    for lr in lr_cats:
        if lr and lr.lower() in text:
            result[1] = lr
            break

    return result


def old_action_TP_Generell(api, tracker, bot_action):
    """
    General extraction action. Expects helpers:
      - old_tp_generell_generate_answer(entities)
      - old_get_lr_text(lr)
    `old_tp_generell_extract_information` should return a sequence like (category, lr)
    """
    entities = old_tp_generell_extract_information(tracker.latest_message["text"])
    try:
        if entities and entities[0]:
            entries = old_tp_generell_generate_answer(entities)
            if not entries:
                message = "Mir sind leider keine " + entities[0]
                if len(entities) > 1 and entities[1]:
                    message += " im Lebensraum " + entities[1]
                message += " bekannt."
                api.bot_utter(message)
            else:
                buttons = []
                random.shuffle(entries)
                it = min(5, len(entries))
                for i in range(it):
                    cur_entry = entries[i]
                    buttons.append(api.create_button(cur_entry, "Was ist ein/e " + cur_entry))
                message = "Hier sind zufällig ausgewählte " + entities[0]
                if len(entities) > 1 and entities[1]:
                    message += " aus dem Lebensraum " + entities[1]
                message += ":"
                api.bot_utter(message, buttons=buttons)
        elif entities and len(entities) > 1 and entities[1]:
            message = old_get_lr_text(entities[1])
            buttons = [
                api.create_button("Tiere (" + entities[1] + ")", "Welche Tiere leben in " + entities[1]),
                api.create_button("Pflanzen (" + entities[1] + ")", "Welche Pflanzen gibt es im Lebensraum " + entities[1])
            ]
            api.bot_utter(message, buttons=buttons)
        else:
            api.bot_utter(
                "Du kannst Fragen zu Tieren und Pflanzen in der Aue stellen. "
                "Frage z.B. 'Welche Tiere gibt es in der Rheinaue?', "
                "'Welche Fische leben im Stillgewässer?' oder "
                "'Welche Pflanzen gibt es in der Hartholzaue?'"
            )
    except Exception as e:
        api.bot_utter("Fehler bei der Verarbeitung der Anfrage.")
        print("action_TP_Generell error:", e)


# lr_texts are now "Erkennungsmerkmale" in tiere_pflanzen_auen for the various types
# so use like so in the future
def old_get_lr_text(auen_name):
    return BotAction.get_entity_features_static(auen_name, "Erkennungsmerkmale")



bot_utters = {
"no_image":"Tut mir leid, ich habe leider kein Bild.",
"tp_unclear":"Ich habe leider nicht verstanden, welches Tier/welche Pflanze du meinst.",
"gen_unclear":"Du kannst Fragen zu Tieren und Pflanzen in der Aue stellen.\nFrage z.B. 'Welche Tiere gibt es in der Rheinaue?','Welche Fische leben im Stillgewässer?' oder 'Welche Pflanzen gibt es in der Hartholzaue?'",
"error":"Da ist leider etwas schiefgelaufen. Entschuldigung"
}


# ----------------------------------------------------------------------
# BotAction class: new implementation
# ----------------------------------------------------------------------

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

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file for action '{self.name}' not found.")
        except Exception as e:
            raise RuntimeError(f"Error loading data file '{path}': {e}")

        # prepare typed subsets
        self.auen = [e for e in self.data if e.get('Typ') == "Auen"]
        self.tiere = [e for e in self.data if e.get('Typ') == "Tier"]
        self.pflanzen = [e for e in self.data if e.get('Typ') == "Pflanze"]

        # collect keys/columns from first record
        first = self.data[0] if self.data else {}
        self.keys = [k for k in first.keys() if not (k.startswith("Name") or k in ['Typ','Gruppe'])]
        print(f"Loaded data for action '{self.name}' with {len(self.data)} entries.")
        print(f"Available keys: {self.keys}")

        self.bot_utters = {
            "no_image":"Tut mir leid, ich habe leider kein Bild.",
            "tp_unclear":"Ich habe leider nicht verstanden, welches Tier/welche Pflanze du meinst.",
            "gen_unclear":"Du kannst Fragen zu Tieren und Pflanzen in der Aue stellen.\nFrage z.B. 'Welche Tiere gibt es in der Rheinaue?','Welche Fische leben im Stillgewässer?' oder 'Welche Pflanzen gibt es in der Hartholzaue?'",
            "error":"Da ist leider etwas schiefgelaufen. Entschuldigung"
        }




    def extract_animal_or_plant(self, user_input):
        """" Try to find an entity matching the user input, first as animal, 
        then as plant.
        """
        ents = self.find_entity(user_input, "Tier")
        if not ents:
            ents = self.find_entity(user_input, "Pflanze")
        return ents

    def tp_generell_extract_information(self, user_input):
        """" Try to find an entity matching the user input, first as animal, 
        then as plant.
        Roughly follow original plan like:
            tp_generell_extract_information(latest_msg):result_matching = process.extractOne(latest_msg, animal_categories)
                if result_matching[1] > 80
                    result[0] = result_matching[0]
                else result_matching = process.extractOne(latest_msg, ["Tiere", "Pflanzen", "B\u00e4ume", "Blumen"])
                    if result_matching[1] > 80
                        if result_matching[0] == "Tiere":result[0] = "Tiere" 
                        else result[0] = "Pflanzen" result_matching = process.extractOne(latest_msg, lr_categories)
                            if result_matching[1] > 80
                                result[1] = result_matching[0]
                                    current_lr = result[1]
                                    return result
        """
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
        searchImage = False
        searchAudio = False

        if "habitat" in key.lower():
            searchKeys_ = ["Lebensraum", "Habitat"]
        elif "lebensraum" in key.lower():
            searchKeys_ = ["Lebensraum", "Habitat"]
        elif "merkmale" in key.lower():
            searchKeys_ = ["Erkennungsmerkmale", "Größe","Gewicht","Verhalten","Lebensweise"]
            searchImage = True
        elif "aussehen" in key.lower():
            searchKeys_ = ["Größe"]
            searchImage = True
        elif "paarung" in key.lower():
            searchKeys_ = ["Paarung", "Fortpflanzung"]
        elif "fortpflanzung" in key.lower():
            searchKeys_ = ["Paarung", "Fortpflanzung"]
        elif ("größe" in key.lower()) or ("groesse" in key.lower()):
            searchKeys_ = ["Größe"]
        elif "lateinischername" in key.lower():
            searchKeys_ = ["Name_sci"]
        elif "rufe" in key.lower():
            searchKeys_ = ["Rufe"]
            searchAudio = True
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
                if searchImage:
                    # also collect image links if available
                    for f in items:
                        if 'Links' in f and f['Links']:
                            for l in f['Links']:
                                img = l.get("img",None)
                                if img:
                                    values["image"].append(img)
                                    break
                if searchAudio:
                    # also collect audio links if available
                    for f in items:
                        if 'Links' in f and f['Links']:
                            for l in f['Links']:
                                audio = l.get("audio",None)
                                if audio:
                                    values["audio"].append(audio)
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
