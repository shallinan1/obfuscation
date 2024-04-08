from unidecode import unidecode
import json
import os

def clean_text(text):
    return unidecode(text)

def save_to_jsonl(data, filename):
    """Saves the data to a JSONL file, creating directories if they don't exist."""
    # Extract the directory part from the filename
    directory = os.path.dirname(filename)
    
    # If the directory part is not empty and does not exist, create it
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Proceed to save the data to the file
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            json_record = json.dumps(item, ensure_ascii=False)
            f.write(json_record + '\n')

def read_from_jsonl(filename):
    """Reads data from a JSONL file and returns a list of items."""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data