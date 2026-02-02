import json
import os
import random
import requests

try:
    from rapidfuzz import process, fuzz, utils
    matchProcessor = utils.default_process
except ImportError:
    # ubuntu 20.04 LTS compatibility
    from fuzzywuzzy import process, fuzz, utils
    matchProcessor = utils.full_process



# ----------------------------------------------------------------------
# Bot utterances and options
# ----------------------------------------------------------------------

bot_utters = {
    "de": {
    "no_image": "Tut mir leid, ich habe leider kein Bild.",
    "tp_unclear": "Ich habe leider nicht verstanden, welches Tier/welche Pflanze du meinst.",
    "gen_unclear": "Du kannst Fragen zu Tieren und Pflanzen in der Aue stellen.\nFrage z.B. 'Welche Tiere gibt es in der Rheinaue?','Welche Fische leben im Stillgewässer?' oder 'Welche Pflanzen gibt es in der Hartholzaue?'",
    "error": "Da ist leider etwas schiefgelaufen. Entschuldigung",
    "measurement_unclear": "Entschuldigung, ich habe nicht verstanden, welchen Messwert du abfragen möchtest.",
    "measurement_prompt": "Welchen Messwert möchtest du wissen? Du kannst mich nach O3 (Ozon), PM10 (Feinstaub Größe 10), PM2,5 (Feinstaub Größe 2,5) oder NO2 (Stickstoff-Dioxid) fragen.",
    "measurement_unavail": "Hm... die Messstation liefert mir aktuell keine Zahlenwerte für diesen Messwert. Versuche es doch später noch einmal!",
    "expertise_prompt": "Hier findest du ein paar ausgewählte Themen, über die ich dir mehr erzählen kann 😃:",
    }
}

bot_measures = {
    "measurement_info": [
        {"text":"Was bedeutet Informationsschwelle?","title":'Informationsschwelle'},
        {"text":"Was bedeutet Alarmschwelle?","title":'Alarmschwelle'},
        {"text":"Woher kommen die Daten?","title":'Messdaten_Woher'}
    ],
    "measurement_eval": {
        "de":{
            "Alarm":{"text":"Alarm: Der Wert liegt über der Alarmschwelle. Ein Besuch im NAZKA wird nicht empfohlen."},
            "Warning":{"text":"Warnung: Der Wert liegt über der Informationsschwelle. Ein Besuch im NAZKA wird nicht empfohlen."},
            "OK":{"text":"Der Wert liegt unter der Informationsschwelle. Ein Besuch im NAZKA ist bendenklos möglich."},
            "LQI":{"text":"Der Luftqualitätsindex (LQI) ist ein zusammengesetzter Index, der die Luftqualität anhand mehrerer Messwerte bewertet. Er berücksichtigt unter anderem Ozon, Feinstaub und Stickstoffdioxid. Die Klassifikation orientiert sich an dem schlechtesten Wert aus diesen Messwerten.","link":{"url":"https://www.lubw.baden-wuerttemberg.de/-/luftqualitatsindex-fur-baden-wurttemberg","titel":"Mehr zum Luftqualitätsindex (LQI)"}},
            "PM2,5":{"text":"Für weitere Informationen zu Feinstaub PM2,5 kannst du dich gerne hier informieren", "link": {"title":"Mehr zu PM2,5","url":"https://www.lubw.baden-wuerttemberg.de/luft/messwerte-immissionswerte?id=DEBW081&comp=7#karte"}},
            "PM10":{"text":"Der Immissionsgrenzwert zum Schutz der menschlichen Gesundheit beträgt 50 µg/m³ (Tagesmittelwert) bei 35 zugelassenen Überschreitungen im Kalenderjahr. Für weitere Informationen zu Feinstaub PM10 kannst du dich gerne hier informieren","link": {"title":"Mehr zu PM10","url":"https://www.lubw.baden-wuerttemberg.de/luft/messwerte-immissionswerte?id=DEBW081&comp=6#karte"}}

            }
        }
}

# BotAction class: new implementation
# ----------------------------------------------------------------------


class BotAction:
    def __init__(self, path, parameters=None):
        self.path = path
        self.name = os.path.basename(path)
        self.parameters = parameters or {}
        try:
            with open(path, "r", encoding="utf-8") as f:
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
        self.auen = [e for e in self.data if e.get("Typ") == "Auen"]
        self.tiere = [e for e in self.data if e.get("Typ") == "Tier"]
        self.pflanzen = [e for e in self.data if e.get("Typ") == "Pflanze"]

        # collect keys/columns from first record
        first = self.data[0] if self.data else {}
        self.keys = [
            k
            for k in first.keys()
            if not (k.startswith("Name") or k in ["Typ", "Gruppe"])
        ]
        print(f"Loaded data for action '{self.name}' with {len(self.data)} entries.")
        print(f"Available keys: {self.keys}")

        self.DEBUG = True
        self.matchThreshold = 75


    def execute(self, handler_name, input=None, context=None, lang="de"):
        """
        Execute a handler function by name with given parameters.
        Returns a tuple (success, result) where success is a boolean and result is the handler output.
        """
        handler_map = {
            "anreise": self.handle_anreise,
            "ausstellung": self.handle_ausstellung,
            "kinder": self.handle_kinder,
            "bio_feature": self.handle_bio_feature,
            "messdaten": self.handle_messdaten,
        }
        
        if handler_name not in handler_map:
            return False, {"error": f"Unknown handler: {handler_name}"}

        if self.DEBUG:
            print(f"Executing handler '{handler_name}' with input: {input} and context: {context}")
        
        try:
            return handler_map[handler_name](input=input, context=context, lang=lang)
        except Exception as e:
            if self.DEBUG:
                print(f"Error executing handler '{handler_name}': {e}")
            return False, {"error": str(e)}

    def getHandlers(self):
        """
        Return a list of available handler names.
        """
        return ["anreise", "ausstellung", "kinder", "bio_feature", "messdaten"]

    def handle_anreise(self, input=None, context=None, lang="de"):
        """
        Handler for 'anreise' action.
        """
        if self.DEBUG:
            print(f"Handling 'anreise' with parameters: {input}")
        # TODO: Implement anreise logic
        return True, {"text": "Anreise information placeholder"}

    def handle_ausstellung(self, input=None, context=None, lang="de"):
        """
        Handler for 'ausstellung' action.
        """
        if self.DEBUG:
            print(f"Handling 'ausstellung' with parameters: {input}")
        # TODO: Implement ausstellung logic
        return True, {"text": "Ausstellung information placeholder"}

    def handle_kinder(self, input=None, context=None, lang="de"):
        """
        Handler for 'kinder' action.
        """
        if self.DEBUG:
            print(f"Handling 'kinder' with parameters: {input}")
        # TODO: Implement kinder logic
        return True, {"text": "Kinder information placeholder"}

    def handle_bio_feature(self, input=None, context=None, lang="de"):
        """
        Handler for 'bio_feature' action.
        Expects parameters like: {'entity': 'entity_name', 'feature': 'feature_key'}
        """
        if self.DEBUG:
            print(f"Handling 'bio_feature' with parameters: {input}")
        
        entity_name = context.get("entity", "") if context else ""
        feature_key = context.get("feature", "") if context else ""
        
        if not entity_name or not feature_key:
            return False, {"error": "Missing entity or feature parameter"}
        
        features = self.get_entity_features(entity_name, feature_key)
        if features:
            return True, features
        else:
            return False, {"text": bot_utters[lang].get("tp_unclear", "No information found")}

    def handle_messdaten(self, input=None, context=None, lang="de"):
        """
        Handler for 'messdaten' action.
        Expects parameters like: {'type': 'O3'|'PM10'|'PM2,5'|'NO2'|'LQI', 'lang': 'de'|'en'|'fr'}
        """
        if self.DEBUG:
            print(f"Handling 'messdaten' with parameters: {input}")
        
        measurement_type = input if input else ""
        
        if not measurement_type:
            return False, {"text": bot_utters[lang].get("measurement_prompt", "")}
        
        return self.measurement_retrieval(measurement_type, debug=self.DEBUG, lang=lang)



    def setDebug(self, debug):
        self.DEBUG = debug

    def setThreshold(self, threshold: int):
        """Set the fuzzy match threshold (0-100)."""
        self.matchThreshold = int(threshold)

    def getThreshold(self):
        """Get the fuzzy match threshold (0-100)."""
        return self.matchThreshold


    @staticmethod
    def measurement_retrieval(type,debug=False,lang="de"):
        """
        Action to fetch and report air quality measurement values.
        Expects `fetch_json_data(url)` helper function to exist.
        """
        messwert = type.upper()
        if debug: print("Requested measurement type:", messwert)

        url = ""
        messwert_label = ""
        unit = " µg/m³."
        time = {"de":" aktuell ", "en":" currently ", "fr":" actuellement "}
        success = True

        lqi = {"de":["sehr gut", "gut", "befriedigend", "ausreichend", "schlecht", "sehr schlecht"],
               "en":["very good", "good", "satisfactory", "sufficient", "poor", "very poor"],
               "fr":["très bien", "bien", "satisfaisant", "suffisant", "pauvre", "très pauvre"]
               }

        if messwert == "O3":
            messwert_label = "Der Ozonwert beträgt"
            url = "https://lupo-cloud.de/air/metric/kit.iai.test.o3"
        elif messwert == "PM10":
            messwert_label = "Der Feinstaubwert PM10 beträgt"
            url = "https://lupo-cloud.de/air/metric/kit.iai.test.pm10"
        elif messwert == "PM2,5":
            messwert_label = "Der Feinstaubwert PM2,5 beträgt"
            url = "https://lupo-cloud.de/air/metric/kit.iai.test.pm25k"
        elif messwert == "NO2":
            messwert_label = "Der Stickstoff-Dioxidwert beträgt"
            url = "https://lupo-cloud.de/air/metric/kit.iai.test.no2"
        elif messwert == "LQI":
            messwert_label = "Nach Luftqualitätsindex ist die Luftqualität"
            url = "https://lupo-cloud.de/air/metric/kit.iai.test.luqx"
        else:
            return False,{"text": bot_utters[lang]["measurement_prompt"]}

        url += "?from=2d-ago&labels=station:DEBW081"
        json_data = requests.get(url).json()
        if len(json_data[0]["values"]) == 0:
            return False,{"text": bot_utters[lang]["measurement_unavail"]}

        if debug: print("Fetched JSON data:", json_data[0]["values"][-5:])
            
        # try to get the most recent valid value. iterate over last 5 values. break at the first valid one.
        options = []
        for k in range(5):
            value = json_data[0]["values"][-(k+1)]
            try:
                value = int(value)
                success = True
            except TypeError:
                success = False
            if success:
                if messwert == "LQI":
                    unit = "."
                    message = messwert_label + time[lang] + " " + lqi[lang][value-1] + "."
                    extension = bot_measures["measurement_eval"][lang]["LQI"]
                    output = {"text": message + extension.get("text",""), "link": extension.get("link", None)}
                else:
                    message = messwert_label + time[lang] + str(value) + unit
                    evaluation = BotAction.measurement_eval(messwert, value ,debug)
                    if debug:
                        print("Evaluation result:", evaluation) 
                    extension = bot_measures["measurement_eval"][lang].get(evaluation, {})
                    output = {"text": message + extension.get("text","")}
                    if extension.get("link") is not None:
                        output["link"] = extension["link"]
                break

        return True, output


    @staticmethod
    def measurement_eval(type,value, debug=False):
        grenzwerte = {"O3":[180, 240], "NO2":[200, 400]}
        # grenzwerte = {"O3":[2,5], "NO2":[200, 400]}  # testing only
        type = type.upper()        
        if type in grenzwerte:
            boundaries = grenzwerte[type]
            if value >= boundaries[1]:
                return "Alarm"
            elif value >= boundaries[0]:
                return "Warning"
            else:
                return "OK"
        elif type == "PM2,5":
            return "PM2,5"
        elif type == "PM10":
            return "PM10"
        else:
            return None

    def extract_animal_or_plant(self, user_input):
        """ " Try to find an entity matching the user input, first as animal,
        then as plant.
        """
        ents = self.find_entity(user_input, "Tier")
        if not ents:
            ents = self.find_entity(user_input, "Pflanze")
        return ents

    def tp_generell_extract_information(self, user_input):
        """ " Try to find an entity matching the user input, first as animal,
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
        if self.DEBUG:
            print(f"Extracting general information for input '{user_input}'")
            
        return self.find_entity(user_input)

    def find_entity(self, user_input, entity_type=None):
        try:
            terms = list(set([user_input] + user_input.split(" ")))
            if self.DEBUG:
                print(f"Finding entity for input '{user_input}' with type '{entity_type}'")
            for term in terms:
                term = term.strip()
                if not term:
                    continue
                if self.DEBUG: 
                    print(f"Searching for term '{term}' and type '{entity_type}'")
                ents = []
                if entity_type is None:
                    targets = [(e.get("Name", "") or "").lower() for e in self.data]
                    if self.DEBUG:
                        print(f"Searching in {len(targets)} total entities.")
                    matches = process.extract(
                        term.lower(),
                        targets,
                        scorer=fuzz.WRatio,
                        limit=5,
                    )
                    if self.DEBUG:
                        print(f"Matches found: {matches}")
                    # limit matches to those above threshold
                    matches = [match for match in matches if match[1] >= self.matchThreshold]
                    try:
                        idxs = [targets.index(match[0]) for match in matches]
                    except:
                        idxs = []
                    if self.DEBUG:
                        print(f"Matches above threshold: {matches}")
                    ents = [self.data[idx] for idx in idxs]
                    #ents = [self.data[match[2]] for match in matches]
                    if self.DEBUG:
                        print(f"Entities found: {ents}")
                        
                    # determine type from first hit and restrict to that type
                    if ents:
                        inferred = ents[0].get("Typ")
                        ents = [e for e in ents if e.get("Typ") == inferred]

                elif entity_type == "Tier":
                    targets = [(e.get("Name", "") or "").lower() for e in self.tiere]
                    if self.DEBUG:
                        print(f"Searching in {len(targets)} total entities.")
                    matches = process.extract(
                        term.lower(),
                        targets,
                        scorer=fuzz.WRatio,
                        limit=5,
                    )
                    if self.DEBUG:
                        print(f"Tiere found: {matches}")
                    matches = [match for match in matches if match[1] >= self.matchThreshold]
                    try:
                        idxs = [targets.index(match[0]) for match in matches]
                    except:
                        idxs = []
                    if self.DEBUG:
                        print(f"Matches above threshold: {matches}")
                    ents = [self.tiere[idx] for idx in idxs]

                elif entity_type == "Pflanze":
                    targets = [(e.get("Name", "") or "").lower() for e in self.pflanzen]
                    if self.DEBUG:
                        print(f"Searching in {len(targets)} total entities.")   
                    matches = process.extract(
                        term.lower(),
                        targets,
                        scorer=fuzz.WRatio,
                        limit=5,
                    )
                    if self.DEBUG:
                        print(f"Pflanzen found: {matches}")
                    # limit matches to those above threshold
                    matches = [match for match in matches if match[1] >= self.matchThreshold]
                    if self.DEBUG:
                        print(f"Matches above threshold: {matches}")
                    try:
                        idxs = [targets.index(match[0]) for match in matches]
                    except:
                        idxs = []
                    if self.DEBUG:
                        print(f"Matches above threshold: {matches}")
                    ents = [self.pflanzen[idx] for idx in idxs]

                elif entity_type == "Auen":
                    targets = [(e.get("Name", "") or "").lower() for e in self.auen]
                    if self.DEBUG:
                        print(f"Searching in {len(targets)} total entities.")
                    matches = process.extract(
                        term.lower(),
                        targets,
                        scorer=fuzz.WRatio,
                        limit=5,
                    )
                    if self.DEBUG:
                        print(f"Auen found: {matches}")
                    # limit matches to those above threshold
                    matches = [match for match in matches if match[1] >= self.matchThreshold]
                    if self.DEBUG:
                        print(f"Matches above threshold: {matches}")
                    try:
                        idxs = [targets.index(match[0]) for match in matches]
                    except:
                        idxs = []
                    if self.DEBUG:
                        print(f"Matches above threshold: {matches}")
                    ents = [self.auen[idx] for idx in idxs]

                if self.DEBUG:
                    if ents and len(ents) > 0:
                        print(
                            f"Found {len(ents)} entities for term '{term}' and type '{entity_type}'"
                        )
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
            if self.DEBUG: print("No matching keys found.")
            return []
        except Exception as e:
            if self.DEBUG: print(f"Error finding keys: {e}")
            return []

    def get_entity_features(self, name, key):
        # check synonyms
        #'Erkennungsmerkmale', 'Habitat', 'Fortpflanzung', 'Größe', 'Links', 'Familie',
        # 'Gattung', 'Lebensraum', 'Klasse', 'Lebensweise', 'Nahrung', 'Feinde',
        # 'Lebenserwartung', 'Schutz', 'Wissenswertes', 'Blütezeit', 'Verwendung',
        # 'Frucht', 'Vorkommen', 'Genießbarkeit', 'Ökologische Bedeutung', 'Giftigkeit',
        # 'Alter', 'Gewicht', 'Überwinterung', 'Verhalten', 'Paarung']

        # key from intent:
        # tp_groesse, tp_habitat, tp_erkennungsmerkmale, tp_lateinischername,
        # tp_lebenserwartung, tp_fortpflanzung, tp_aussehen
        # tp_definition is basically all together
        searchImage = False
        searchAudio = False

        if "habitat" in key.lower():
            searchKeys_ = ["Lebensraum", "Habitat"]
        elif "definition" in key.lower():
            # everything
            searchKeys_ = self.keys
            searchImage = True
            searchAudio = True
        elif "lebensraum" in key.lower():
            searchKeys_ = ["Lebensraum", "Habitat"]
        elif "merkmale" in key.lower():
            searchKeys_ = [
                "Erkennungsmerkmale",
                "Größe",
                "Gewicht",
                "Verhalten",
                "Lebensweise",
            ]
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

        if self.DEBUG:
            print(f"Getting features for entity '{name}' with key '{key}' mapped to search keys '{searchKeys}'")
            
        try:
            items = self.find_entity(name)
            if self.DEBUG:
                print("Items found:", len(items))
            if items:
                values = {"text": [], "image": [], "audio": []} #, "video": [], "link": []}
                for key in searchKeys:
                    if key == "Links":
                        continue  # skip Links here
                    values["text"].extend(
                        [f.get(key) for f in items if key in f and f.get(key)]
                    )
                if searchImage:
                    # also collect image links if available
                    for f in items:
                        if "Links" in f and f["Links"]:
                            for l in f["Links"]:
                                img = l.get("img", None)
                                if img and img != "":
                                    values["image"].append(img)
                                    break
                if searchAudio:
                    # also collect audio links if available
                    for f in items:
                        if "Links" in f and f["Links"]:
                            for l in f["Links"]:
                                audio = l.get("audio", None)
                                if audio and audio != "":
                                    values["audio"].append(audio)
                                    break
                if self.DEBUG:
                    print(
                        f"Found features for entity '{name}', keys '{searchKeys}':",
                        values,"Text:",values["text"]
                    )
                if values["text"] == [] and values["image"] == [] and values["audio"] == []:
                    #return {}
                    values["text"] = ["Zu dieser Eigenschaft sind für " + name + " leider keine Informationen vorhanden."]
                return values
            else:
                print("No matching entity found for features:", name)
                return {}
        except Exception as e:
            print(f"Error getting features: {e}")
            return {}

if __name__ == "__main__":
    action = BotAction("../rawData/tiere_pflanzen_auen.json")
    for user_input in [
        "frosch habitat",
        "fisch",
        "blume",
        "wasserfrosch",
        "auen",
        "magerrasen",
    ]:
        result = action.extract_animal_or_plant(user_input)
        if result:
            print("1:", [r.get("Name") for r in result])
            for f in ["Lebensraum", "Merkmale", "Aussehen"]:
                name = result[0].get("Name")
                features = action.get_entity_features(name, f)
                if features:
                    print(f"1: {name}   Feature '{f}':", features)

        else:
            print("1: No results found.\n-----\n")

        result = action.find_entity(user_input, entity_type="Tier")
        if result:
            print("2:", [r.get("Name") for r in result])
        else:
            print("2: No results found.\n-----\n")

        result = action.tp_generell_extract_information(user_input)
        if result:
            print("3: Type detected:", result[0].get("Typ"))
            print("3:", [r.get("Name") for r in result])
        else:
            print("3: No results found.\n-----\n")
        print("-----\n")

        result = action.find_entity_key(user_input)
        if result:
            print("4:", result)
        else:
            print("4: No results found.\n-----\n")
