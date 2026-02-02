import json
import sys

def limit_array_length(data, max_length=5):
    """Recursively limit the length of all arrays to max_length."""
    if isinstance(data, dict):
        # Iterate through dictionary keys and apply to each value
        for key, value in data.items():
            data[key] = limit_array_length(value, max_length)
    elif isinstance(data, list):
        # If it's a list, limit its length
        data = data[:max_length]
        # Apply recursively to any elements that are lists or dicts
        for i in range(len(data)):
            data[i] = limit_array_length(data[i], max_length)
    return data

# Example usage
json_file_path = sys.argv[1] # "your_file.json"  # Replace with your file path

# Load the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Traverse and limit arrays to length 5
data = limit_array_length(data, max_length=3)

# If you want to print the modified data
print(json.dumps(data, indent=2))

# Optionally, save the modified data back to a file
with open('modified_file.json', 'w') as file:
    json.dump(data, file, indent=2)
