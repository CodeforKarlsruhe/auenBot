import json
import pandas as pd
import os

class BotAction:
    def __init__(self, name, parameters=None):
        self.name = name
        self.parameters = parameters if parameters is not None else {}  
        try:
            self.data = pd.read_json(f'../rawData/{name}.json')  # Load entity data from JSON file
            self.auen = self.data[self.data.Typ == "Auen"]
            self.tiere = self.data[self.data.Typ == "Tier"]
            self.pflanzen = self.data[self.data.Typ == "Pflanze"]
            self.keys = [k for k in self.data.keys() if not (k.startswith("Name") or k in ['Typ','Gruppe'])]
            print(f"Loaded data for action '{name}' with {len(self.data)} entries.")
            print(f"Available keys: {self.keys}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file for action '{name}' not found.")
    
    def extract_animal_or_plant(self, user_input):
        # Placeholder for actual extraction logic
        # This function would contain the logic to extract animal or plant information from user input
        extracted_info = self.find_entity(user_input,"Tier")  # Example extracted data
        if extracted_info.empty:
            extracted_info = self.find_entity(user_input,"Pflanze")
        return extracted_info
    def tp_generell_extract_information(self, user_input):
        # Placeholder for actual extraction logic
        # This function would contain the logic to extract general information from user input
        extracted_info = self.find_entity(user_input)  # Example extracted data
        return extracted_info
    
    def find_entity(self, user_input, entity_type=None):
        # Placeholder for actual entity finding logic
        # This function would contain the logic to find an entity based on type and name
        # full input search first
        try:
            for term in [user_input] + user_input.split(" "):
                if entity_type is None:
                    ents = self.data[self.data.Name.str.contains(term, case=False, na=False)]
                    if not ents.empty:
                        entity_type = ents.iloc[0]['Typ']
                        ents = ents[ents.Typ == entity_type]
                elif entity_type == "Tier":
                    ents = self.tiere[self.tiere.Name.str.contains(term, case=False, na=False)]
                elif entity_type == "Pflanze":
                    ents = self.pflanzen[self.pflanzen.Name.str.contains(term, case=False, na=False)]
                elif entity_type == "Auen":
                    ents = self.auen[self.auen.Name.str.contains(term, case=False, na=False)]
                if not ents.empty:
                    return ents
                else:
                    print("No matching entity found for:", term)
            return pd.DataFrame([])
        except Exception as e:
            print(f"Error finding entity: {e}")
            return pd.DataFrame([])


    def find_key(self, user_input):
        # Placeholder for actual entity finding logic
        # This function would contain the logic to find an entity based on type and name
        try:
            for term in [user_input] + user_input.split(" "):
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
        if not result.empty:
            print("1:",[r for r in result.Name.to_list()])
        else:
            print("1: No results found.\n-----\n")
        result = action.find_entity(user_input, entity_type="Tier")
        if not result.empty:
            print("2:",[r for r in result.Name.to_list()])
        else:
            print("2: No results found.\n-----\n")
        result = action.tp_generell_extract_information(user_input)
        if not result.empty:
            print("3: Type detected:", result.iloc[0]['Typ'])
            print("3:",[r for r in result.Name.to_list()])    
        else:
            print("3: No results found.\n-----\n")
        print("-----\n")
        result = action.find_key(user_input)
        if result:
            print("4:", result)
        else:
            print("4: No results found.\n-----\n")
