import json
import os

class BotAction:
    def __init__(self, name, parameters=None):
        self.name = name
        self.parameters = parameters or {}
        path = os.path.join('..', 'rawData', f'{name}.json')
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
            print(f"Loaded data for action '{name}' with {len(self.data)} entries.")
            print(f"Available keys: {self.keys}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file for action '{name}' not found.")
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
                if entity_type is None:
                    ents = [e for e in self.data if term.lower() in (e.get('Name') or '').lower()]
                    if ents:
                        # determine type from first hit and restrict to that type
                        inferred = ents[0].get('Typ')
                        ents = [e for e in ents if e.get('Typ') == inferred]
                elif entity_type == "Tier":
                    ents = [e for e in self.tiere if term.lower() in (e.get('Name') or '').lower()]
                elif entity_type == "Pflanze":
                    ents = [e for e in self.pflanzen if term.lower() in (e.get('Name') or '').lower()]
                elif entity_type == "Auen":
                    ents = [e for e in self.auen if term.lower() in (e.get('Name') or '').lower()]
                else:
                    ents = [e for e in self.data if term.lower() in (e.get('Name') or '').lower() and e.get('Typ') == entity_type]

                if ents:
                    return ents
                else:
                    print("No matching entity found for:", term)
            return []
        except Exception as e:
            print(f"Error finding entity: {e}")
            return []

    def find_key(self, user_input):
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


if __name__ == "__main__":
    action = BotAction("tiere_pflanzen_auen")
    for user_input in ["frosch habitat","fisch", "blume", "wasserfrosch", "auen","magerrasen"]:
        result = action.extract_animal_or_plant(user_input)
        if result:
            print("1:", [r.get('Name') for r in result])
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

        result = action.find_key(user_input)
        if result:
            print("4:", result)
        else:
            print("4: No results found.\n-----\n")
