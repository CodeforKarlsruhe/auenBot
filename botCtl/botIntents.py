import json
import os

try:
    from rapidfuzz import process, fuzz, utils
    matchProcessor = utils.default_process
except ImportError:
    # ubuntu 20.04 LTS compatibility
    from fuzzywuzzy import process, fuzz, utils
    matchProcessor = utils.full_process

import random

from botActions import BotAction


# Context schema (shared static JSON) ------------------------------------------------
DEFAULT_CONTEXT_SCHEMA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), ".", "botSchema.json")
)

def load_context_schema(path=None):
    """Load and return the context JSON schema from path or default location.
    Returns an empty dict if the file is missing. Raises on JSON parse errors.
    """
    p = path or DEFAULT_CONTEXT_SCHEMA_PATH
    try:
        with open(p, "r", encoding="utf-8") as f:
            schema = json.load(f)
        context = schema.get("properties", {}).get("context", {}).get("properties", {})
        return context
        
    except FileNotFoundError:
        return {}
    except Exception as e:
        raise RuntimeError(f"Error loading context schema '{p}': {e}")

# module-level schema object for import by other modules
CONTEXT_SCHEMA = load_context_schema()


INTENT_MESSAGES = {
    "de": {
        "no_intent": "Kein Intent im Kontext angegeben.",
        "intent_not_found": "Intent nicht gefunden.",
        "handler_not_found": "Handler nicht gefunden.",
        "options_file_not_found": "Fehler: Optionsdatei für Completion nicht gefunden.",
        "no_input_completion": "Fehler: Kein Eingabewert für Completion.",
        "no_items_file": "Fehler: Keine Items-Datei für Completion-Intent vorhanden.",
        "no_match_completion": "Fehler: Eingabe hat kein Item für Completion gefunden.",
        "no_input_options": "Fehler: Kontext hat Optionen, aber keine Eingabe angegeben.",
        "invalid_key": "Fehler: Ungültiger Schlüssel.",
        "select_options": "Bitte wählen Sie eine der folgenden Optionen:",
        "completed": "Abgeschlossen.",
        "continue_completion": "Bitte fahren Sie mit der Vervollständigung fort."
    },
    "en": {
        "no_intent": "No intent specified in context.",
        "intent_not_found": "Intent not found.",
        "handler_not_found": "Handler not found.",
        "options_file_not_found": "Error: Options file not found for completion.",
        "no_input_completion": "Error: No input for completion.",
        "no_items_file": "Error: No items file available for completion intent.",
        "no_match_completion": "Error: Input did not match any item for completion.",
        "no_input_options": "Error: Context has options but no input provided.",
        "invalid_key": "Error: Invalid key.",
        "select_options": "Please select one of the following options:",
        "completed": "Completed.",
        "continue_completion": "Please continue with the completion."
    },
    "fr": {
        "no_intent": "Aucune intention spécifiée dans le contexte.",
        "intent_not_found": "Intention non trouvée.",
        "handler_not_found": "Gestionnaire non trouvé.",
        "options_file_not_found": "Erreur : Fichier d'options introuvable pour la complétion.",
        "no_input_completion": "Erreur : Aucune entrée pour la complétion.",
        "no_items_file": "Erreur : Aucun fichier d'éléments disponible pour l'intention de complétion.",
        "no_match_completion": "Erreur : L'entrée n'a fait correspondre aucun élément pour la complétion.",
        "no_input_options": "Erreur : Le contexte a des options mais aucune entrée fournie.",
        "invalid_key": "Erreur : Clé invalide.",
        "select_options": "Veuillez sélectionner l'une des options suivantes :",
        "completed": "Terminé.",
        "continue_completion": "Veuillez continuer avec la complétion."
    },
}


__all__ = ["CONTEXT_SCHEMA", "load_context_schema", "DEFAULT_CONTEXT_SCHEMA_PATH", "INTENT_MESSAGES"]

class BotIntent:
    """
    BotIntent
    ----------
    Loads a JSON intent definition file and provides simple handlers to execute
    intents. The class is small and intentionally conservative: it expects a file
    that contains either a list of intent dictionaries or a dictionary of
    dictionaries (values are intent dicts). A single intent dict is also
    accepted.

    Usage:
      intent = BotIntent("/path/to/intents.json")
      intent.setDebug(True)
      result = intent.execute("default", intent_name="greet", context={}, input="hi")

    Public methods:
      - setDebug(bool)
      - get_intent_by_id(id) -> dict|None
      - get_intent_by_name(name) -> dict|None
      - execute(handler="default", intent_name=None, input=None, context=None, lang="de")
    """

    def __init__(self, path, parameters=None):
        self.DEBUG = False
        self.path = path
        self.name = os.path.basename(path)
        self.parameters = parameters or {}
        self.actionCtl = None
        self.actionNames = []
        self.matchThreshold = 75  # default fuzzy match threshold

        # load data file (accept list, dict-of-dicts or single dict)
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            if isinstance(loaded, list):
                self.data = loaded
            elif isinstance(loaded, dict):
                # if values are dicts, treat it as a dict-of-records
                if all(isinstance(v, dict) for v in loaded.values()):
                    self.data = list(loaded.values())
                else:
                    # fallback: wrap single dict as single record list
                    self.data = [loaded]
            else:
                self.data = []

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file for action '{self.name}' not found.")
        except Exception as e:
            raise RuntimeError(f"Error loading data file '{path}': {e}")

        # named handlers available to execute()
        self.handlers = {
            "default": self.__defaultHandler,
            "complete": self.__completeHandler,
        }

    @staticmethod
    def getMsg(key, lang = "de"):
        """Get a localized message by language code and key."""
        if not lang or lang not in INTENT_MESSAGES:
            lang = "de"
        msgs = INTENT_MESSAGES.get(lang, INTENT_MESSAGES["de"])
        return msgs.get(key, f"{msgs.get('invalid_key', '')} ({key})")

    @staticmethod
    def dummyAction(name, input=None, lang="de"):
        """A dummy action function that can be used as a handler."""
        if name is None or name.strip() == "":
            return {"text": "Error: Unnamed actions cannot be called!"}
        output = {}
        if input is None or input.strip() == "":
            output["text"] = f"Dummy action '{name}' executed with no input."
        else:
            opts = random.sample(["text","link","options"], k=random.randint(1,3))
            input = input.strip() + " " + " ".join(opts)
            if "text" in input.lower():
                output["text"] = f"Dummy action '{name}' received text input: {input}"
            if "link" in input.lower():
                output["link"] = {"title":"Testlink", "url":f"http://example.com/dummy_action/{name}"}
            if "options" in input.lower():
                output["options"] = [
                    {"title": f"Option 1 for {name}", "value": "opt1"},
                    {"title": f"Option 2 for {name}", "value": "opt2"},
                    {"title": f"Option 3 for {name}", "value": "opt3"},
                ]
        return output

    def setDebug(self, debug: bool):
        """Enable or disable debug printing."""
        self.DEBUG = bool(debug)
        if self.actionCtl is not None:
            self.actionCtl.setDebug(self.DEBUG)

    def setThreshold(self, threshold: int):
        """Set the fuzzy match threshold (0-100)."""
        self.matchThreshold = int(threshold)
        if self.actionCtl is not None:
            self.actionCtl.setThreshold(self.matchThreshold)

    def getThreshold(self):
        """Get the fuzzy match threshold (0-100)."""
        return self.matchThreshold

    def setActions(self, definition_file):
        """Set a BotActions instance to use for action processing."""
        self.actionCtl = BotAction(definition_file)
        self.actionNames = self.actionCtl.getHandlers()
        if self.DEBUG:
            print(f"BotIntent: Loaded actions from '{definition_file}':", self.actionNames)

    def get_intent_by_id(self, intent_id):
        """Return the intent dict with matching 'id' or None if not found."""
        for entry in self.data:
            if entry.get("id") == intent_id:
                return entry
        return None

    def get_intent_by_name(self, intent_name):
        """Return the intent dict with matching 'intent' or None if not found."""
        for entry in self.data:
            if entry.get("intent") == intent_name:
                return entry
        return None

    def execute(self, intent_name=None, input=None, context=None, lang="de"):
        """
        Execute the handler of an intent.

        - intent_name: name of intent to execute (must be present in loaded data)
        - input: user input string (may be None)
        - context: dict representing current context; if None an empty dict is used
        - lang: language suffix used to pick utterances (e.g. 'de' -> 'utter_de')

        Returns: dict with keys "output" and "context".
        """
        if self.DEBUG:
            print(f"Executing handler for intent '{intent_name}'")

        if intent_name is None or intent_name.strip() == "":
            raise ValueError("No intent name specified for execution.")
        intent_name = intent_name.strip()
        intent = self.get_intent_by_name(intent_name)
        if intent is None:
            raise ValueError(f"Intent '{intent_name}' not found.")
        handler = intent.get("handler", "default")
        if self.DEBUG:
            print(f"Using handler '{handler}' for intent '{intent_name}'")

        # resolve handler (accept callables for flexibility)
        if callable(handler):
            handler_func = handler
        else:
            handler_func = self.handlers.get(handler)
            if handler_func is None:
                raise ValueError(f"Handler '{handler}' not found.")

        # find intent
        intent = self.get_intent_by_name(intent_name)
        if self.DEBUG:
            print(f"Resolved intent for '{intent_name}':", intent)
        if intent is None:
            raise ValueError(f"Intent '{intent_name}' not found.")

        # ensure we don't accidentally mutate a caller's context reference
        ctx = dict(context) if context is not None else {}
        result = handler_func(intent, input=input, context=ctx, lang=lang)

        if self.DEBUG:
            print(f"Handler '{handler}' result:", result)
        return result

    def _make_error_response(self, msg_key, context, intent, lang):
        """
        Create an error response with localized message.
        
        Args:
            msg_key: Message key from INTENT_MESSAGES
            context: Current context dict
            intent: Intent dict
            lang: Language code
            
        Returns:
            dict with "output" and "context" keys
        """
        if self.DEBUG:
            print(f"_make_error_response: Creating error response for key '{msg_key}'")
        message = BotIntent.getMsg(msg_key, lang)
        # always remove options and target on error. leave rest
        # add error info to context?
        ctx = dict(context)
        ctx.pop("options", None)
        ctx.pop("target", None)
        #ctx["last_intent"] = intent.get("intent")
        #ctx["intent"] = None
        return {"output": {"text": message}, "context": ctx, "error": msg_key}

    def _load_options_file(self, option_name):
        """
        Load options from data/options_{option_name}.json.
        
        Args:
            option_name: Name of the options file (without prefix/suffix)
            
        Returns:
            List of option dicts or None if file not found
        """
        if self.DEBUG:
            print(f"_load_options_file: Loading options for '{option_name}'")
        try:
            options_path = os.path.join(os.path.dirname(__file__), "data", f"options_{option_name}.json")
            with open(options_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            if self.DEBUG:
                print(f"Options file not found for '{option_name}'")
            return None

    def _load_items_file(self, requirement, intent_base):
        """
        Load items from data/{requirement}_{intent_base}.json.
        
        Args:
            requirement: Requirement name
            intent_base: Base name of the intent
            
        Returns:
            List of items or None if file not found
        """
        if self.DEBUG:
            print(f"_load_items_file: Loading items for requirement '{requirement}' and intent base '{intent_base}'")
        try:
            items_path = os.path.join(os.path.dirname(__file__), "data", f"{requirement}_{intent_base}.json")
            with open(items_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            if self.DEBUG:
                print(f"Items file not found for requirement '{requirement}' and intent base '{intent_base}'")
            return None

    def _find_first_missing_requirement(self, requirements, context):
        """
        Find the first requirement that is not present in context.
        
        Args:
            requirements: List of requirement names
            context: Current context dict
            
        Returns:
            First missing requirement name or None if all satisfied
        """
        if self.DEBUG:
            print(f"_find_first_missing_requirement: Checking requirements {requirements} against context keys {list(context.keys())}")
        for req in requirements:
            if req not in context:
                return req
        return None

    def _match_input_to_items(self, input_text, items):
        """
        Match user input against a list of items using fuzzy matching.
        
        Args:
            input_text: User input string
            items: List of item strings
            threshold: Minimum score for a valid match (default: 75)
            
        Returns:
            Matched item string or None if no match above threshold
        """
        if self.DEBUG:
            print(f"_match_input_to_items: Matching input '{input_text}' against items")
        choices = [str(i) for i in items]
        if self.DEBUG:
            print(f"Matching input '{input_text}' against items (count={len(choices)})")

        match = process.extractOne(input_text.lower(), [c.lower() for c in choices], 
                                   scorer=fuzz.WRatio, processor=matchProcessor)
        if not match:
            if self.DEBUG:
                print("No match found (extractOne returned None).")
            return None

        # fuzzywuzzy returns (match, score), rapidfuzz returns (match, score, index)
        if len(match) > 2:
            _, score, idx = match
            term = choices[idx]
        else:
            term_, score = match
            term = [t for t in choices if t.lower() == term_.lower()][0]
            
        if self.DEBUG:
            print("Match result:", match, term)
        
        if score >= self.matchThreshold:
            return term
        
        if self.DEBUG:
            print(f"Input '{input_text}' did not match any item (score {score}).")
        return None

    def _match_input_to_options(self, input_text, options):
        """
        Match user input against a list of option dicts.
        
        Args:
            input_text: User input string
            options: List of option dicts with "title" key
            
        Returns:
            Matched option title or None if no match
        """
        if self.DEBUG:
            print(f"_match_input_to_options: Matching input '{input_text}' against options")
        option_titles = [o.get("title", "").lower() for o in options]
        if self.DEBUG:
            print("Options: ", option_titles)

        return self._match_input_to_items(input_text, option_titles)

    def _process_action(self, intent, input_text, context={}, lang="de"):
        """
        Process the action defined in an intent.
        
        Args:
            intent: Intent dict
            input_text: User input string
            lang: Language code
            
        Returns:
            Action result dict or empty dict if no action
        """
        if self.DEBUG:
            print(f"_process_action: Processing action for intent '{intent.get('intent')}'")
        action = intent.get("action", None)
        if self.DEBUG:
            print("Action to process:", action)
        
        if action is not None and action.strip() != "":
            # check if action is defined in actionCtl
            if self.actionCtl is not None and action in self.actionNames:
                if self.DEBUG:
                    print(f"Executing action '{action}' via BotActions.")
                input = input_text or ""
                status, action_result = self.actionCtl.execute(action, input=input, context=context, lang=lang)
                if self.DEBUG:
                    print("Action result:", status, action_result)
                return action_result
            # fallback: use dummy action
            else:
                if self.DEBUG:
                    print(f"Executing dummy action '{action}'.")            
                input = input_text or ""
                action_result = BotIntent.dummyAction(action, input=",".join([input,json.dumps(list(context.keys()))]), lang=lang)
                if self.DEBUG:
                    print("Action result:", action_result)
                return action_result


    def _update_context_with_provides(self, intent, context):
        """
        Update context with values from intent's "provides" field.
        
        Args:
            intent: Intent dict
            context: Context dict to update (modified in-place)
        """
        if self.DEBUG:
            print(f"_update_context_with_provides: Updating context with provides from intent '{intent.get('intent')}'")
        provides = intent.get("provides") or {}
        if self.DEBUG:
            print("Provided:", provides)
        for key, value in provides.items():
            if self.DEBUG:
                print(f"Updating context with provided '{key}': '{value}'")
            context[key] = value

    def _finalize_completion(self, intent, context, requirements, input_text, lang):
        """
        Finalize a completed intent by processing action and cleaning up context.
        
        Args:
            intent: Intent dict
            context: Current context dict
            requirements: List of requirement names to remove from context
            input_text: User input string
            lang: Language code
            
        Returns:
            dict with "output" and "context" keys
        """
        if self.DEBUG:
            print(f"_finalize_completion: Finalizing completion for intent '{intent.get('intent')}'")

        ctx = dict(context)
        ctx.pop("options", None)
        # insert input as last_input in context
        # context["last_input"] = input_text

        # Check for target routing
        if "target" in requirements and "target" in context:
            target_name = context.get("target")
            matched_intent = self.get_intent_by_name(target_name)
            if matched_intent is None:
                if self.DEBUG:
                    print(f"Target intent '{target_name}' not found.")
                return self.__make_error_response("intent_not_found", ctx, intent, lang)

            if self.DEBUG:
                print(f"Routing to matched intent '{target_name}' -> {matched_intent}")
            
            output = {"text": f"{BotIntent.getMsg('completed', lang)} Route to intent: {matched_intent.get('intent') if matched_intent else target_name}"}
            # overwrite intent with target
            ctx = { "intent": target_name, "last_input": input_text }
            if self.DEBUG:
                print(f"Executing routed intent '{target_name}'")
            return self.execute(intent_name=target_name, context=ctx, input=input_text, lang=lang)

        else:
            # Process action or use default completion message
            action_result = self._process_action(intent, input_text, ctx, lang)
            output = action_result if action_result else {"text": BotIntent.getMsg("completed", lang)}

        # Remove requirements from context
        for req in requirements:
            ctx.pop(req, None)

        return {"output": output, "context": ctx,"completed": True}

    def __defaultHandler(self, intent, input=None, context=None, lang="de"):
        """Simple handler that returns the configured utterance and optionally
        processes a placeholder action. Does not perform completion or matching."""
        if self.DEBUG:
            print("Default handler for intent:", intent.get("intent"), "lang:", lang)
            print("Input:", input)
            print("Context (in):", context)

        context = context or {}

        locale = "" if lang is None else f"_{lang}"
        text = intent.get(f"utter{locale}", "") or ""
        output = {"text": text}

        if intent.get("link") is not None:
            output["link"] = intent.get("link")

        # Process action if present
        action_result = self._process_action(intent, input, context, lang)
        if action_result:
            for key, value in action_result.items():
                if key != "text":
                    output[key] = value
                else:
                    output[key] = (output[key] + " " + value).strip()

        # insert input as last_input in context
        # context["last_input"] = input
        return {"output": output, "context": context,"completed": True}

    def __completeHandler(self, intent, input=None, context=None, lang="de"):
        """
        Handler for completion-style intents. Manages multi-step interactions
        by presenting options or matching input against required items.
        """
        if self.DEBUG:
            print("Complete handler for intent:", intent.get("intent"), "lang:", lang)
            print("Input:", input)
            print("Context (in):", context)

        context = context or {}
        intent_base = (intent.get("intent", "").split("_")[0] or "").strip()
        
        if self.DEBUG:
            print("Intent base:", intent_base)

        # Update context with provided values
        self._update_context_with_provides(intent, context)

        # Parse requirements
        reqs = [r.strip() for r in (intent.get("requires") or "").split(",") if r.strip() != ""]
        if self.DEBUG:
            print("Intent requires:", reqs)

        # Handle options presentation or input matching
        ctx_opts = context.get("options")
        # insert input as last_input in context
        # context["last_input"] = input
        
        if ctx_opts is None:
            return self._handle_no_options(intent, input, context, reqs, intent_base, lang)
        else:
            return self._handle_with_options(intent, input, context, ctx_opts, reqs, lang)

    def _handle_no_options(self, intent, input_text, context, requirements, intent_base, lang):
        """
        Handle completion when no options are currently in context.
        Either present options or match input against items.
        """
        if self.DEBUG:
            print(f"_handle_no_options: No options in context")
            print("Intent:", intent)
        # Present options if configured
        option_name = (intent.get("options") or "").strip()
        if self.DEBUG:
            print("Option name from intent:", option_name)
        if option_name:
            if self.DEBUG:
                print("Presenting options for completion:", option_name)
            
            options = self._load_options_file(option_name)
            if options is None:
                return self._make_error_response("options_file_not_found", context, intent, lang)

            ctx = dict(context)
            ctx["options"] = options
            ctx["intent"] = intent.get("intent")
            
            output = {"text": BotIntent.getMsg("select_options", lang)}
            if intent.get("link") is not None:
                output["link"] = intent.get("link")
            
            return {"output": output, "context": ctx}
            # ########## completed #############

        # No options configured - match input against items
        if not input_text or not input_text.strip():
            return self._make_error_response("no_input_completion", context, intent, lang)

        # Find first missing requirement and its items
        missing_req = self._find_first_missing_requirement(requirements, context)
        if self.DEBUG:
            print("First missing requirement:", missing_req)
            
        if missing_req is None:
            if self.DEBUG:
                print("All requirements satisfied.")
            return self._finalize_completion(intent, context, requirements, input_text, lang)
            # ########## completed #############
        

        items = self._load_items_file(missing_req, intent_base)
        if items is None:
            return self._make_error_response("no_items_file", context, intent, lang)

        # Match input to items
        matched_value = self._match_input_to_items(input_text, items)
        if matched_value is None:
            return self._make_error_response("Nichts gefunden... hier könnte ein Vorschlag kommen", context, intent, lang)

        # Update context and finalize
        context[missing_req] = matched_value
        if self.DEBUG:
            print(f"Updating context with '{missing_req}': '{matched_value}'")

        # Check if all requirements satisfied
        all_satisfied = all(req in context for req in requirements)
        if self.DEBUG:
            print("Requirements satisfied?:", all_satisfied, "Requirements:", requirements, "Context keys:", list(context.keys()))
        
        if all_satisfied:
            if self.DEBUG:
                print("All requirements satisfied.")
            return self._finalize_completion(intent, context, requirements, input_text, lang)
        else:
            if self.DEBUG:
                print("Some requirements still missing.")
            # Not completed - continue without options
            ctx = dict(context)
            ctx["intent"] = intent.get("intent")
            
            return {"output": {"text": BotIntent.getMsg("continue_completion", lang)}, "context": ctx}

    def _handle_with_options(self, intent, input_text, context, options, requirements, lang):
        """
        Handle completion when options are present in context.
        Match selected option and update requirements.
        """
        if self.DEBUG:
            print(f"_handle_with_options: Options present in context")
        if not input_text or not input_text.strip():
            return self._make_error_response("no_input_completion", context, intent, lang)

        # Match input to options
        matched = self._match_input_to_options(input_text, options)
        if matched:
            if self.DEBUG:
                print("Matched")
            # Fill first missing requirement
            missing_req = self._find_first_missing_requirement(requirements, context)
            if missing_req:
                context[missing_req] = matched
                if self.DEBUG:
                    print(f"Updating context with '{missing_req}': '{matched}'")
        else:
            if self.DEBUG:
                print("Not matched")
            return self._make_error_response("no_match_completion", context, intent, lang)
        
        # Check if all requirements satisfied
        all_satisfied = all(req in context for req in requirements)
        if self.DEBUG:
            print("Requirements satisfied?:", all_satisfied, "Requirements:", requirements, "Context keys:", list(context.keys()))
        
        if all_satisfied:
            if self.DEBUG:
                print("All requirements satisfied.")
            return self._finalize_completion(intent, context, requirements, input_text, lang)
        else:
            if self.DEBUG:
                print("Some requirements still missing.")
            # Not completed - continue with options
            ctx = dict(context)
            ctx["options"] = options
            ctx["intent"] = intent.get("intent")
            
            return {"output": {"text": BotIntent.getMsg("continue_completion", lang)}, "context": ctx}
    



if __name__ == "__main__":
    import sys
    intent = BotIntent("../rawData/intents.json")
    #intent = BotIntent("./data/test_intents.json")
    intent.setActions("../rawData/tiere_pflanzen_auen.json")
    if len(sys.argv) > 1 and sys.argv[1] == "-d":
        intent.setDebug(True)
    lang = "de"
    print(f"Loaded intent '{intent.name}' with {len(intent.data)} entries.")
    for i in intent.data:
        print(i["intent"],i["id"])
        result = intent.execute(i["intent"], context={"intent": i["intent"]}, lang=lang)
        print("Result:", result)
        ctx = result.get("context",{})
        if "options" in ctx:
            print("  -> Options:", [o["title"] for o in ctx["options"]])
            print("\nSimulate selection of an option...")
            result = intent.execute(i["intent"], context=result["context"], input=random.choice(ctx["options"])["title"], lang=lang)
            print("Result2:", result)
        else:
            ireqs = intent.get_intent_by_name(i["intent"]).get("requires","") or ""
            ireqs = [r.strip() for r in ireqs.split(",") if r.strip() != ""]
            reqs = [r for r in ctx.keys() if r in ireqs]
            for r in ireqs:
                print(f"  -> Requirement '{r}'")
                if r not in reqs:
                    print("\nSimulate input for requirement...")
                    # load items file
                    items = intent._load_items_file(r, i["intent"].split("_")[0])
                    if items:
                        simulated_input = random.choice(items)
                        print(f"Simulated input: {simulated_input}")
                        result = intent.execute(i["intent"], context=result["context"], input=simulated_input, lang=lang)
                        print("Result2:", result)
                    else:
                        print(f"No items file for requirement '{r}' and intent base '{i['intent'].split('_')[0]}'")
                        result = intent.execute(i["intent"], context=result["context"], input="abxx", lang=lang)
                        print("Result2:", result)

        print("\n\n")


    exit()
        
    # Intent matching test (expertise)
    print("\n--- Intent matching Test ---")
    test_intent = "test8"

    result = intent.execute(test_intent, input=None, context={"intent": test_intent}, lang=lang)
    print("Result with no input:", result)        
    print("\n\n")
    result = intent.execute(test_intent, input="test1", context=result["context"], lang=lang)
    print("Result with input 'test1':", result)        
    print("\n\n")
    result = intent.execute(test_intent, input="xxx", context={"intent": test_intent}, lang=lang)
    print("Result with input 'xxx':", result)
    print("\n\n")
    
    print("\n--- Feature matching Test ---")
    test_intent = "test2"

    result = intent.execute( test_intent, input=None, context={"intent": test_intent}, lang=lang)
    print("Result with no input:", result)        
    print("\n\n")
    result = intent.execute( test_intent, input="Feat2_4", context=result["context"], lang=lang)
    print("Result with input 'Feat2_4':", result)        
    print("\n\n")
    result = intent.execute( test_intent, input="feat3", context={"intent": test_intent}, lang=lang)
    print("Result with input 'feat3':", result)
    print("\n\n")
    result = intent.execute( test_intent, input="feaxx", context={"intent": test_intent}, lang=lang)
    print("Result with input 'feaxx':", result)
    print("\n\n")
    
    print("\n--- Feature matching Test with ent,type ---")
    test_intent = "test2b"
    test_handler = intent.get_intent_by_name(test_intent)["handler"]

    result = intent.execute( test_intent, input=None, context={"intent": test_intent}, lang=lang)
    print("Result with no input:", result)        
    print("\n\n")
    result = intent.execute( test_intent, input="Feat2_4", context=result["context"], lang=lang)
    print("Result with input 'Feat2_4':", result)        
    print("\n\n")
    result = intent.execute( test_intent, input="feat3", context={"intent": test_intent}, lang=lang)
    print("Result with input 'feat3':", result)
    print("\n\n")
    result = intent.execute( test_intent, input="feaxx", context={"intent": test_intent}, lang=lang)
    print("Result with input 'feaxx':", result)
    print("\n\n")
    
    