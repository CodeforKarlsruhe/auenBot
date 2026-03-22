# Description of sample files

## raw_intents.json

list of intent items.

### Format

>      {
        "id": "64709aa78ef7ef3b45d4fa79",
        "intent": "tiere_gewicht",
        "utter": null,
        "link": null,
        "action": "bio_feature",
        "alias": "Fragen zum Gewicht von Tieren",
        "handler": "complete",
        "requires": "entity,feature",
        "options": null,
        "provides": {
            "feature": "gewicht",
            "type": "Tiere"
        }
    },


### Keys
*id* and *intent* are required and unique. intent is the human readable equivalent to id
*utter*, optional. If present, this is the direct textual response to this intent. Only for simple intents. Null for complex intents.
*link*, optional. A url to be used in the response.
*action*, optional. Agent action to be performed when responding, e.g. reading environmental data.
*alias*, required. Extended textual description of the intent.
*handler*, required. Handler function for complex intents, when keys *options* or *requires* are non zero. Typically a q&a sequence must be performede
*requires*, optional. List of q&a slot identifiers for this intent, e.g. entity or feature
*options*, optional. Name of a file which contains list of options to be used in q&a sequence
*provides*, optional. Slot items provided already by this intent. NB: may provide slots which are not required.

### Sample items

1 simple intent ("63b6a1f6d9d1941218c5c7c4") with direct response and 3 complex intents


## intent_texts.json

Sample sentences for the 4 intents. Can be used a references in vector search.

## data.json

One sample entity dataset of the 3 available types *Tier*, *Pflanze*, *Auen*. Keys *name* and *name_sci* are mandatory,
all others keys are optional.


# General Operation

  1. User input
  2. Intent decoder: find hat the user want. E.g. similarity vector search against intent texts
  3. If intent with direct response: Respond with utter => Finished
  4. If complex intent: slot processing: find required slots (e.g. type, entity, feature) from intent. Find provided slots. Q&A to complete missing slots.
  5. If slots completed: extract entity/feature data from database or get *action* result for utterance => Finished


