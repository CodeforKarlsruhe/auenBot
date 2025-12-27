# AuenBot
Migration from LUBW/NAZKA ChatBot KarlA


# Status

Tried to analyze dataset from LUBW

Extracted chatbot signatures for intents and responses 

Extracted data from plants, animals and Rheinauen" area

Intents also relate to access to current whether conditions and environmental data. 
Need to be implemented (probably fairly straightforward)


## RawData

  * pflanzenKeys.json: Parameters for plant descriptions 
  * tiereKeys.json: Parameters for animal descriptions 
  * tiere_pflanzen_auen.json: dataset for animals, plants and some Rheinauen types. Each items has a 
  *type* parameter.
  * taskList.json: decoded signatures. if *utter* is present, it should be used as response. Otherwise, intent should either start with *tp_*, *tiere_*, *pflanzen_*  which should then address the data from the corresponding types (or both), or with *wetter* or *messdaten*.  Reference to the few *Rheinauen* datasets has to be defined still.


## Next Steps
### Basic Bot
Create vector embedding for all intent texts. Setup database with vectors, full text and intent names. Test chatbot response to arbitrary requests. 

### Reference Data

Add data access to whether conditions, environmental data, Wikidata images and audio files, Source to be found, probably from NAZKA, or https://www.museumfuernaturkunde.berlin/de/forschung/tierstimmenarchiv. MP3 files were missing in input dataset.

