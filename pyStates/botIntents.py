import json
import os

class BotIntent:
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

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file for action '{name}' not found.")
        except Exception as e:
            raise RuntimeError(f"Error loading data file '{path}': {e}")


if __name__ == "__main__":
    intent = BotIntent("intents_translated")
    print(f"Loaded intent '{intent.name}' with {len(intent.data)} entries.")
    for i in intent.data[:5]:
        print(i["intent_de"])
    