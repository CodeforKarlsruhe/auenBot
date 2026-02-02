import json
from pathlib import Path
from typing import Any, Dict
import os

from openpyxl import DEBUG

from botLlm import OpenAICompatClient
from botVectors import load_vectors, query_vectors


system_prompt_check_intent_de = """Du bist ein Intent‑Klassifizierungssystem für einen Chatbot.
    "Dir werden Fragen zu Tieren, Pflanzen und natürlichen Lebensräumen in den Karlsruher Rheinauen gestellt.
    "Ein bestimmtes Biotop wird im Deutschen ‚Aue‘ genannt.
    "Für den Chatbot sind mehrere Intents definiert.
    "Basierend auf der Benutzereingabe wählen den am besten passenden Intent aus den bereitgestellten Optionen aus.
    "Beachten Sie, dass Verweise auf Tiere oder Pflanzen in der Regel nicht mit Ernährung, sondern mit biologischen Aspekten zusammenhängen.
    "Wenn keiner passt, gebe als Index -1 zurück. Antworte nur mit dem Index. Gibt keinen weiteren Text zurück.
    "Die aktuelle Benutzersprache ist Deutsch."""

system_prompt_check_intent_en = """You are an intent classification system for a chatbot. 
                        "You will be asked questions about animals, plants and natural habitats in the Karlsruher Rheinauen.
                        "A particular biotiope is called 'Aue' in German. There are a number of intents defined for the chatbot. 
                        "Given the user input, select the best matching intent from the provided options. 
                        "Note that typically reference to animals or plants are not related to nutrition but to biological aspects. 
                        "If none match, respond with 'None'.
                        "The current user language is English."""



class BotDecoder:
    """_summary_

    Raises:
        ValidationError: _description_

    Returns:
        _type_: _description_
    """

    def __init__(self):
        self.DEBUG = False
        self.intents = None
        self.vectors = None
        self.vector_intents = None
        self.llm = None
        self.DEBUG = False
        self.thresholds = {"low": 0.25, "high": 0.75}
        self.neighbors = 5

        # load private config for llm
        try:
            import private as pr  # type: ignore

            self.private = {
                "apiKey": getattr(pr, "apiKey", None),
                "baseUrl": getattr(pr, "baseUrl", None),
                "embUrl": getattr(pr, "embUrl", None),
                "embMdl": getattr(pr, "embMdl", None),
                "lngMdl": getattr(pr, "lngMdl", None),
            }
            print("Loaded private config for LLM.")
        except Exception:
            print("No private config found for LLM.")
            self.private = None
            raise RuntimeError("No private config found for LLM.")

    def __get_intent_by_id(self, intent_id):
        """Return the intent dict with matching 'id' or None if not found."""
        for entry in self.intents:
            if entry.get("id") == intent_id:
                return entry
        return None

    def setDebug(self,debug):
        self.DEBUG = debug
        if self.llm:
            self.llm.setDebug(debug)

    def getDebug(self):
        return self.DEBUG

    def setNeighbors(self, n: int):
        self.neighbors = n
        if self.DEBUG:
            print(f"Neighbors set to: {n}")
            
    def getNeighbors(self):
        return self.neighbors

    def setThresholds(self, low: float, high: float):
        self.thresholds["low"] = low
        self.thresholds["high"] = high
        if self.DEBUG:
            print(f"Thresholds set to low: {low}, high: {high}")

    def getThresholds(self):
        return self.thresholds

    def loadModels(self):
        api_key = self.private.get("apiKey")
        base_url = self.private.get("baseUrl")
        emb_url = self.private.get("embUrl", base_url)
        embed_model = self.private.get("embMdl")
        chat_model = self.private.get("lngMdl")
        llm = OpenAICompatClient(
            base_url=base_url,
            api_key=api_key,
            emb_url=emb_url,
            chat_model=chat_model,
            embed_model=embed_model,
        )
        if self.DEBUG:
            print(f"LLM Client initialized with model {llm.chat_model} / {llm.embed_model}")

        self.llm = llm

    def getModels(self):
        return {
            "base_url": self.llm.base_url,
            "emb_url": self.llm.emb_url,
            "chat_model": self.llm.chat_model,
            "embed_model": self.llm.embed_model,
        }

    def loadIntents(self,intent_path: str):
        # load intent file
        try:
            with open(intent_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            if isinstance(loaded, list):
                self.intents = loaded
            elif isinstance(loaded, dict):
                # if values are dicts, treat it as a dict-of-records
                if all(isinstance(v, dict) for v in loaded.values()):
                    self.intents = list(loaded.values())
                else:
                    # fallback: wrap single dict as single record list
                    self.intents = [loaded]
            else:
                self.intents = []

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file for intents '{intent_path}' not found.")
        except Exception as e:
            raise RuntimeError(f"Error loading data file '{intent_path}': {e}")

        if self.DEBUG:
            print("Intents with actions loaded: ", self.intents)

    def loadVectors(self,vectors_path: str):
        self.vectors, self.vector_intents = load_vectors(vectors_path)
        if self.DEBUG:
            print(f"Loaded {len(self.vectors)} intent vectors from {vectors_path}.")


    # ------------------------------------------------------
    # intent detection function
    # ------------------------------------------------------
    def detectIntent(self,user_input: str, lang="de") -> tuple[str | list, bool]:
        search = self.llm.embed([user_input])
        # print("Input embedding:", search[0])
        candidates = query_vectors(self.vectors, search[0],self.neighbors)
        if self.DEBUG: print("Intent candidates:", candidates)
        if not candidates or len(candidates[0]) == 0:
            # no intent found, return error
            fallback = self.__get_intent_by_id("63b6a1f6d9d1941218c5c7d2")
            target_intent = fallback["intent"]
            if self.DEBUG: print("No candidates found, using fallback")
            return target_intent, False

        # find best intent    
        best_intent_idx = candidates[0][0]  # ["intent_id"]
        best_intent_id = self.vector_intents[best_intent_idx]
        best_score = candidates[1][0].astype(float)  # ["intent_id"]
        best_intent = self.__get_intent_by_id(best_intent_id)
        if self.DEBUG: print(
            f"Best intent id: {best_intent_id}, intent: {best_intent}, score: {best_score}"
        )

        # very low confidence, use fallback
        if best_score <= self.thresholds["low"]:
            if self.DEBUG: print("Low score: Using fallback")
            fallback = self.__get_intent_by_id("63b6a1f6d9d1941218c5c7d2")
            target_intent = fallback["intent"]
            return target_intent, False

        # high confidence
        elif best_score >= self.thresholds["high"]:
            target_intent = best_intent.get("intent", None)
            if self.DEBUG: print("Selected high score best intent:", best_intent)
            return target_intent, False

        # intermediate confidence. check with LLM
        else:
            # medium confidence, return options
            # first scan through candidate list. keep only the first candidate (with the best score) for each intent_id
            intent_options = []
            intent_aliases = []
            seen = set()
            for i in range(0, len(candidates[0])):
                idx = candidates[0][i]
                intent_id = self.vector_intents[idx]
                score = candidates[1][i].astype(float)
                intent_name = self.__get_intent_by_id(intent_id).get("intent")
                intent_alias = self.__get_intent_by_id(intent_id).get(f"alias_{lang}", None)
                if not intent_alias or intent_alias == "":
                    intent_alias = intent_name
                if self.DEBUG: print(f" Next intent: intent: {intent_name}, score: {score}")
                if intent_name in seen:
                    if self.DEBUG: print("Skipping duplicate intent:", intent_name)
                    continue
                seen.add(intent_name)
                intent_options.append(intent_name)
                intent_aliases.append(intent_alias)

            if self.DEBUG:
                print("Intent aliases:",intent_aliases)

            # check if only one left after deduplication. this must be the best one already
            if len(intent_options) == 1:
                target_intent = best_intent.get("intent", None)
                if self.DEBUG: print(
                    "Selected lower score best intent after deduping:",
                    target_intent
                )
                return target_intent, False
                    
            else:
                if self.DEBUG: print("Remaining intent options:", intent_options)
                options = []
                for o in range(len(intent_options)):
                    options.append({"title": intent_options[o],"text":intent_aliases[o]})
                
                if self.DEBUG: print("Call llm to find better intent ...")
                # we have the aliases already 
                if self.DEBUG: print("Intent options with alias:", intent_aliases)

                try:
                    llmResult = self.llm.chat_json(
                        temperature=0.0,
                        system=system_prompt_check_intent_de,
                        user=f"Nutzereingabe: '{user_input}'. "
                        f"Verfügbare Intents: {', '.join(intent_aliases)}. ",
                    )
                    if self.DEBUG: print("LLM intent result:", llmResult)
                except:
                    if self.DEBUG: print("LLM execution failed")
                    llmResult = -1

                if llmResult is None:
                    llmResult = -1

                if isinstance(llmResult, str):
                    llmResult = int(llmResult.strip())

                if (
                    isinstance(llmResult, int)
                    and llmResult >= 0
                    and llmResult < len(candidates[0])
                ):
                    idx = candidates[0][llmResult]
                    intent_id = self.vector_intents[idx]
                    best_intent = self.__get_intent_by_id(intent_id)
                    if self.DEBUG: print("Selected high score best intent from LLM:", best_intent)
                    target_intent = best_intent.get("intent", None)
                    if self.DEBUG: print("Selected intent from LLM:", target_intent)
                    # done with llm: intent found
                    return target_intent, True

                else:
                    # need to notify user to select intent
                    if self.DEBUG: print("No valid LLM result, asking user to select intent")
                    return options, True

    def __checkOptions(self,input_text: str, options: list) -> int | None:
        """Check if the input_text matches one of the options.
        Returns the index of the matched option, or None if no match.
        """
        input_lower = input_text.lower()
        for idx, option in enumerate(options):
            if option.lower() in input_lower:
                return idx
        return None


    def intentOptions(self, input_text: str, options: list) -> str:
        """Check if the input_text matches one of the options.
        Returns the index of the matched option, or None if no match.
        """
        if self.DEBUG: print("Checking user input for intent options:", input_text, options)
        selected_idx = self.__checkOptions(input_text, [opt["title"] for opt in options])
        if selected_idx is not None:
            if self.DEBUG: print("User selected option index:", selected_idx)
            selection = options[selected_idx]
            if self.DEBUG: print("User selected option:", selection)

            # check if we are missing intent or entity here 
            # use title as intent first
            if "title" in selection:
                user_intent = selection["title"]
                if self.DEBUG: print("Mapped selected option to intent:", user_intent)
                return True, user_intent

        # not found. clear options and continue
        if self.DEBUG: print("No option matched user input yet, start over")
        fallback = self.intents.get_intent_by_id("63b6a1f6d9d1941218c5c7d2")
        target_intent = fallback["intent"]
        if self.DEBUG: print("Using fallback")
        return False, target_intent        



# ----------------------------------------------------------------------
# 7️⃣ Run the app (development mode)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Running BotDecoder in development mode.")
    decoder = BotDecoder()
    decoder.setDebug(True)
    decoder.loadModels()
    decoder.loadIntents(os.path.join("data","intents_raw.json"))
    decoder.loadVectors(os.path.join("../rawData/","intent_vectors.json"))
    decoder.detectIntent("Hallo Auenbot", lang="de")
