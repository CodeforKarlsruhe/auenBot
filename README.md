# AuenBot
Migration from LUBW/NAZKA ChatBot KarlA


# Status

## Current (2026-02-02)

### Front-End
Companion frontent: https://github.com/CodeforKarlsruhe/auenlaend 

### Backend

This repo: https://github.com/CodeforKarlsruhe/auenBot

Required definition files are in folder **rawData**

 * intents.json: List of intents with text examples, (optional) output and optional properties (to be documented).
 * tiere_pflanzen_auen.json: The entity database 
 * intent_vectors.json: Embedding vectors


All functionality is in folder **botCtl**

  * botCtl.py: the wrapper to run the backend app. External exposure is via apache proxy settings 
  * botDecoder.py: finds target intent or fallback from user input and defined intents
  * botIntents.py: Handles specific intent via handlers and derived/detected properties. Eventual matching is done via *rapidfuzz/fuzzywuzzy* string matching (as in previous implementaion)
  * botActions.py: handles any actions needed for intent output generation, e.g. retrive measurement values from LUBW api
  * botVectors.py: loads and queries intent vectors using LLM ebedding model *bge-m3*. **This is different to the previous implementation**


The script *botCtl/makeDataFiles.py* is used to derive a more compact intent description file (without the text samples) in botCtl/data/intents_raw.json and a series of files with feature and entity lists for matching. 

**NB** The options_\<something\>.json files in the folder *botCtl/data* cannot be generated automatically but must be manually edited (as in repo).

**botCtl.py** handles the frontend communication and stores a record of received and transmitted messages in *bot_history.db* (sqlite).


### Localization
Is partially prepared for, defaults to German (de).


### Specifications 

 * Intents: list of intent items like so:
 >   {
    "id":"63b6a1f6d9d1941218c5c7c4",
    "intent":"begrüßung",
    "utter":"Hi, ich bin KarlA, deine Chatbot-Rangerin😊!\nIch freue mich darauf, mit dir zu chatten.<br \/>Als Chatbot-Rangerin beantworte ich Fragen rund um den Lebensraum \"Karlsruher Rheinauen\". Auch Fragen zum Naturschutzzentrum Karlsruhe-Rappenwört kann ich beantworten.<br \/>\nEin paar Hinweise, bevor es losgeht: In der Natur begegnen wir uns alle auf Augenhöhe, daher benutze ich das \"Du\". Ich bitte dich, mir keine persönlichen Daten wie deinen Namen oder deine Adresse zu verraten. Wenn du mehr über mich wissen möchtest: Klicke auf Chatbot-Rangerin KarlA2.\nSo - jetzt ab zu spannenden Naturthemen! Hast du eine Frage oder darf ich dir etwas zu meinen Lieblingsthemen erzählen?",
    "intent_en":"greeting",
    "intent_fr":"bienvenue",
    "intent_de":"begrüßung",
    "utter_de":"Hi, ich bin KarlA, deine Chatbot-Rangerin😊!\nIch freue mich darauf, mit dir zu chatten.<br \/>Als Chatbot-Rangerin beantworte ich Fragen rund um den Lebensraum \"Karlsruher Rheinauen\". Auch Fragen zum Naturschutzzentrum Karlsruhe-Rappenwört kann ich beantworten.<br \/>\nEin paar Hinweise, bevor es losgeht: In der Natur begegnen wir uns alle auf Augenhöhe, daher benutze ich das \"Du\".\n  Ich bitte dich, mir keine persönlichen Daten wie deinen Namen oder deine Adresse zu verraten. Wenn du mehr über mich wissen möchtest: Klicke auf Chatbot-Rangerin KarlA2.\nSo - jetzt ab zu spannenden Naturthemen! Hast du eine Frage oder darf ich dir etwas zu meinen Lieblingsthemen erzählen?",
    "utter_en":"Hi, I'm KarlA, your Chatbot Ranger! 😊 I'm looking forward to chatting with you. As a Chatbot Ranger, I answer questions about the habitat \"Karlsruher Rheinauen\" and also about the Nature Conservation Center in Karlsruhe-Rappenwört.\nA few pointers before we get started: In nature, we all meet on equal footing, so I use the \"Du\".\nI ask you not to reveal any personal data such as your name or address. If you want to know more about me, click here: Chatbot Rangerin KarlA2.\nSo - off to exciting nature topics! Do you have a question or should I tell you something about my favorite topics?",
    "utter_fr":"Bonjour, je suis KarlA, votre guide chatbot pour le site \"Karlsruher Rheinauen\" ! Je suis impatiente de discuter avec vous. En tant que guide chatbot, je répondrai à toutes vos questions sur le site du \"Karlsruher Rheinauen\" et sur le centre de conservation de Karlsruhe-Rappenwört.\nQuelques conseils avant de commencer : dans la nature, nous sommes tous sur un pied d'égalité, c'est pourquoi je vais vous parler du \"tu\". N'oubliez pas de ne pas partager vos informations personnelles comme votre nom ou votre adresse. Si vous voulez en savoir plus sur moi, rendez-vous sur Chatbot-Rangerin KarlA2.\nAlors, plongeons dans des sujets naturels passionnants ! Avez-vous une question ou souhaite-t-il que je vous parle de mes sujets préférés ?",
    "link":{
      "title":"Chatbot-Rangerin KarlA2",
      "url":"https:\/\/nazka.de\/chatbot-karla"
    },
    "action":null,
    "alias_de":null,
    "alias_en":null,
    "alias_fr":null,
    "handler":"default",
    "requires":null,
    "options":null,
    "provides":null
  },

*id* and *intent* are the main keys to identify an intent. 
*Aliases* are descriptive texts for LLM prompting in case LLM resolution of target intent is required.  
*handler* specifies the handling mechanism, currently **default** for a simple intent which just produces an output. Or **complete** for intents which need to assemble property values (entity, feature, type) from user input or provided values. 

Example: 
>   "requires":"entity,feature",
    "options":null,
    "provides":{
      "feature":"definition"
    }

The completion handler requires *entity* and *feature*. From the intent decoding we already get the feature *definition*. So we need to search only for a matching *entity* in the user input. This allows to use generic handling for a number of intents instead of individual handlers for each intent. More (complex) handlers can easily be added, if required.  




 * Database: list of items like so:
 >   {
    "Erkennungsmerkmale":"Die etwa 28mm große Blauschwarze Holzbiene ist unsere größte heimische Wildbienenart. Unter ihrer Last neigt sich mancher Blütenkopf nach unten.  Durch ihren blauschimmernden Panzer und die blauschwarzen Flügel ist sie nicht mit Hummeln zu verwechseln.",
    "Habitat":"Die wärmeliebende Wildbienenart wird durch die Klimaerwärmung in Deutschland immer häufiger.\nSie bewohnt sonnenbeschienene Biotope mit reichlich Totholz und großer Blütenvielfalt. Beliebte Lebensräume der Blauen Holzbiene sind strukturreiche Streuobstwiesen, naturnahe Gärten, Parkanlagen und lichte Waldränder.",
    "Fortpflanzung":"Die Holzbiene lebt nicht in einem Volk, sondern alleine. Sie wird schon früh im Jahr aktiv und beginnt mit ihren kräftigen Mundwerkzeugen eine fingerdicke, etwa 30 cm lange Brutröhre in totes Holz zu nagen. Ans Ende der Röhre trägt sie einen Nahrungsvorrat aus Pollen und Nektar ein, bis die Kammer randvoll ist und legt ein Ei darauf. Anschließend verschließt sie die Zelle mit einer Wand aus Holzspänen und Speichel. In einer Brutröhre werden um die 15 Zellen angelegt. Die fertig entwickelten Bienen fliegen bereits im Juni aus.",
    "Größe":"Die etwa 28mm große Blauschwarze Holzbiene ist unsere größte heimische Wildbienenart. Unter ihrer Last neigt sich mancher Blütenkopf nach unten.",
    "Name":"Blauschwarze Holzbiene",
    "Name_alt":"Große Blaue Holzbiene",
    "Links":[
      {
        "img":"https:\/\/upload.wikimedia.org\/wikipedia\/commons\/thumb\/c\/c6\/Abeille_charpentiere_1024.jpg\/330px-Abeille_charpentiere_1024.jpg"
      }
    ],
    "Typ":"Tier",
    "Name_sci":"Xylocopa violacea",
    "Name_eng":"carpenter bee",
    "Familie":"Echte Bienen (Apidae)",
    "Gattung":"Holzbienen (Xylocopa)",
    "Lebensraum":"Hartholzaue",
    "Klasse":"Insekten",
    "Lebensweise":null,
    "Nahrung":null,
    "Feinde":null,
    "Lebenserwartung":null,
    "Schutz":null,
    "Wissenswertes":null,
    "Blütezeit":null,
    "Verwendung":null,
    "Frucht":null,
    "Vorkommen":null,
    "Genießbarkeit":null,
    "Ökologische Bedeutung":null,
    "Giftigkeit":null,
    "Alter":null,
    "Gewicht":null,
    "Überwinterung":null,
    "Verhalten":null,
    "Paarung":null,
    "Gruppe":"Blauschwarze Holzbiene",
    "Unterklasse":"Hautflügler"
  },

Every items has the keys *Typ*, *Name* and *Name_sci*, all other keys are optional.
So far, *Name* is used for matching.



 * Intent Vectors
    Each intent comes with a list of (German) sample texts. The embedding model is used to generate a vector file **rawData/intent_vectors.json** which created the embedding vector for each sentence and maps it to the corresponding intent id. The vectors can be queried for similarity to find the best N matches for any input. **NB** This is a very simple approach. Brute force search against all vectors and a very large vector file (needs git lfs !). Production would use more efficient implementations (e.g. protobuf to get smaller file). However, with less than 10000 vectors the use of a vector database or appoximate nearist-neightbour search (like faiss) would be overkill (if not already in place).




## Initial (2025-12-30)
Analyzed dataset from LUBW

Extracted chatbot signatures for intents and responses 

Extracted data from plants, animals and Rheinauen" area

Intents also relate to access to current whether conditions and environmental data. 

Initial decoding and routing implemented in python. Input matching with rapidfuzz (text matchin library). 
Fallback to remote LLM if required.
Options for vector-search but initial version with BM25 not very helpfull. Future test with bge-m3 pending (vectors from intent samples already generated).


## RawData

### Primary files
  * tiere_pflanzen_auen.json: knowledge base. dataset for animals, plants and some Rheinauen types.
  * intents.json: intents with sample texts and utterance (if any)
  * intent_vectors.json: vectorized (bge-m3 embeddings) text samples and corresponding intent_id. **Needs git lfs**

### Original bot
  * tagsAndInstructions.json: additional info for original bot decoding and routing

### Intent detection

  1) Find "simple" intents, which ,ap to one specific answer, like *wo gibt es etwas zu essen?*
  2) Find "simple" intents which require function calling, like *wie ist der CO2 Gehalt?*
  3) Find intents which require specific information, usually acquired during multiple steps using a more or less complex context. 
    * Travel information with from,to,date
    * Entity (Tiere, Pflanzen, Auen) information using various key parameters (properties, class, type), like so (from original):

```python
    > tp_generell_extract_information(latest_msg):result_matching = process.extractOne(latest_msg, animal_categories)
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
                      
  def tp_generell_generate_answer(entities):
  result = []
  if entities[0] == "Tiere":
      for animal in ANIMAL.keys():
          if not entities[1] or lr_categories.index(entities[1]) in ANIMAL[animal][4]:
              result.append(animal)
          elif entities[0] == "Pflanzen"
              for plant in PLANT.keys()
                  if not entities[1] or lr_categories.index(entities[1]) in PLANT[plant][4]:
                      result.append(plant)
                  else:
                      for animal in ANIMAL.keys()
                          if ANIMAL[animal][3] == str(animal_categories.index(entities[0])):
                              if not entities[1] or lr_categories.index(entities[1]) in ANIMAL[animal][4]:
                                  result.append(animal)
  return result

```



### Auxiliary, input or leftover files
  * pflanzenKeys.json: Parameters for plant descriptions 
  * tiereKeys.json: Parameters for animal descriptions 
  * taskList.json: decoded signatures. if *utter* is present, it should be used as response. Otherwise, intent should either start with *tp_*, *tiere_*, *pflanzen_*  which should then address the data from the corresponding types (or both), or with *wetter* or *messdaten*.  Reference to the few *Rheinauen* datasets has to be defined still.

## Media

### Directory of fantasy images
512*512, generated by flux-1-schnell.



## Next Steps
### Basic Bot
Create vector embedding for all intent texts. Setup database with vectors, full text and intent names. Test chatbot response to arbitrary requests. 

### Reference Data

Add data access to whether conditions, environmental data, Wikidata images and audio files, Source to be found, probably from NAZKA, or https://www.museumfuernaturkunde.berlin/de/forschung/tierstimmenarchiv. MP3 files were missing in input dataset.

