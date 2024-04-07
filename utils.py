from unidecode import unidecode
import json

def clean_text(text):
    return unidecode(text)

def save_to_jsonl(data, filename):
    """Saves the data to a JSONL file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            json_record = json.dumps(item, ensure_ascii=False)
            f.write(json_record + '\n')
