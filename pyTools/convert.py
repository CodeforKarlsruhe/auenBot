
import json
import re
from collections import defaultdict

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
    
# Save combined results to a single JSON file (remove duplicates while preserving order)
def unique_preserve_order(items):
    """Return items with duplicates removed while preserving order.
    Items are keyed by a JSON-stable serialization when possible to handle dict/list elements."""
    seen = set()
    unique = []
    for item in items:
        try:
            key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        except TypeError:
            # Fallback for objects that json can't serialize (use str representation)
            key = str(item)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


# Function to extract relevant fields from intents and projects
def extract_relevant_fields(data):
    intents = []
    queries = []
    actions = []
    slot_mappings = []
    stories = []

    # Loop through documents and extract relevant fields
    for item in data.get('documents', {}).get('Intent', []):
        # Extract intent related data
        if 'name' in item:
            intents.append(item['name'])
        if 'description' in item:
            queries.append(item['description'])

    # Extract actions (e.g., templates, utterances)
    for action in data.get('documents', {}).get('Action', []):
        if 'name' in action:
            actions.append(action['name'])
        if 'values' in action:
            actions.extend(action['values'])

    # Extract slot mappings (used for matching user inputs)
    for slot in data.get('documents', {}).get('Slot', []):
        if 'name' in slot:
            slot_mappings.append(slot['name'])

    # Extract story data (if present in project file)
    for story in data.get('documents', {}).get('Story', []):
        if 'name' in story:
            stories.append(story['name'])

    return intents, queries, actions, slot_mappings, stories




# Function to extract relevant fields and group by hash values
def extract_and_group_by_hash(data):
    hash_pattern = r'\b[a-f0-9]{24}\b'  # Regex pattern to match 24-character hashes
    grouped_data = defaultdict(lambda: {'intents': [], 'queries': [], 'actions': [], 'slot_mappings': [], 'stories': []})

    # Loop through the data and extract relevant fields
    for item in data.get('documents', {}).get('Intent', []):
        # Extract the hash from the item (search all string fields)
        hash_matches = re.findall(hash_pattern, json.dumps(item))  # Search within the serialized item
        for hash_value in hash_matches:
            # Group data by the hash value
            if 'name' in item:
                grouped_data[hash_value]['intents'].append(item['name'])
            if 'description' in item:
                grouped_data[hash_value]['queries'].append(item['description'])
    
    # Extract actions and slot mappings
    for action in data.get('documents', {}).get('Action', []):
        hash_matches = re.findall(hash_pattern, json.dumps(action))
        for hash_value in hash_matches:
            if 'name' in action:
                grouped_data[hash_value]['actions'].append(action['name'])
            if 'values' in action:
                grouped_data[hash_value]['actions'].extend(action['values'])
    
    # Extract slot mappings
    for slot in data.get('documents', {}).get('Slot', []):
        hash_matches = re.findall(hash_pattern, json.dumps(slot))
        for hash_value in hash_matches:
            if 'name' in slot:
                grouped_data[hash_value]['slot_mappings'].append(slot['name'])
    
    # Extract stories
    for story in data.get('documents', {}).get('Story', []):
        hash_matches = re.findall(hash_pattern, json.dumps(story))
        for hash_value in hash_matches:
            if 'name' in story:
                grouped_data[hash_value]['stories'].append(story['name'])

    return grouped_data

# Example usage
# Example usage
intents_data = load_json("Dialoge_Intents_Antworten 19.11.25/database-version-dump.json")  # Replace with actual file path
projects_data = load_json("Projektdateien KarlA 12.11.25/database-version-dump.json")  # Replace with actual file path


# Extract relevant data
intents, queries, actions, slot_mappings, stories = extract_relevant_fields(intents_data)
project_intents, project_queries, project_actions, project_slot_mappings, project_stories = extract_relevant_fields(projects_data)

# Combine data from both files
all_intents = intents + project_intents
all_queries = queries + project_queries
all_actions = actions + project_actions
all_slot_mappings = slot_mappings + project_slot_mappings
all_stories = stories + project_stories

def _key_for(item):
    try:
        return json.dumps(item, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return str(item)

def _group_pair(a_list, b_list):
    a_map, a_order = {}, []
    b_map, b_order = {}, []
    for it in a_list:
        k = _key_for(it)
        if k not in a_map:
            a_map[k] = it
            a_order.append(k)
    for it in b_list:
        k = _key_for(it)
        if k not in b_map:
            b_map[k] = it
            b_order.append(k)
    # default = items present in both, preserve first-seen order from a_list then b_list
    default = []
    seen = set()
    for it in a_list + b_list:
        k = _key_for(it)
        if k in a_map and k in b_map and k not in seen:
            default.append(a_map[k])
            seen.add(k)
    only_a = [a_map[k] for k in a_order if k not in b_map]
    only_b = [b_map[k] for k in b_order if k not in a_map]
    return default, only_a, only_b


output_path = "combined_results.json"
with open(output_path, "w", encoding="utf-8") as out_file:
    default_intents, only_intents, only_project_intents = _group_pair(intents, project_intents)
    default_queries, only_queries, only_project_queries = _group_pair(queries, project_queries)
    default_actions, only_actions, only_project_actions = _group_pair(actions, project_actions)
    default_slot_mappings, only_slot_mappings, only_project_slot_mappings = _group_pair(slot_mappings, project_slot_mappings)
    default_stories, only_stories, only_project_stories = _group_pair(stories, project_stories)

    json.dump({
        "intents": {
            "default": default_intents,
            "intent": only_intents,
            "project_intent": only_project_intents
        },
        "queries": {
            "default": default_queries,
            "query": only_queries,
            "project_query": only_project_queries
        },
        "actions": {
            "default": default_actions,
            "action": only_actions,
            "project_action": only_project_actions
        },
        "slot_mappings": {
            "default": default_slot_mappings,
            "slot_mapping": only_slot_mappings,
            "project_slot_mapping": only_project_slot_mappings
        },
        "stories": {
            "default": default_stories,
            "story": only_stories,
            "project_story": only_project_stories
        }
    }, out_file, ensure_ascii=False, indent=2)

print(f"Saved combined results to {output_path}")

###################

# Extract and group by hash
grouped_intents_data = extract_and_group_by_hash(intents_data)
grouped_projects_data = extract_and_group_by_hash(projects_data)

# Combine data from both files
combined_grouped_data = defaultdict(lambda: {'intents': [], 'queries': [], 'actions': [], 'slot_mappings': [], 'stories': []})

# Merge grouped data
for hash_value, data in grouped_intents_data.items():
    for key, value in data.items():
        combined_grouped_data[hash_value][key].extend(value)

for hash_value, data in grouped_projects_data.items():
    for key, value in data.items():
        combined_grouped_data[hash_value][key].extend(value)

# Print the grouped data by hash
for hash_value, data in combined_grouped_data.items():
    print(f"Hash: {hash_value}")
    print(f"Intents: {data['intents']}")
    print(f"Queries: {data['queries']}")
    print(f"Actions: {data['actions']}")
    print(f"Slot Mappings: {data['slot_mappings']}")
    print(f"Stories: {data['stories']}")
    print("="*50)



output_path = "combined_grouped_results.json"
with open(output_path, "w", encoding="utf-8") as out_file:
    json.dump(combined_grouped_data, out_file, ensure_ascii=False, indent=2)

print(f"Saved combined results to {output_path}")