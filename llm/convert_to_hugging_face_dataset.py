import json
from datasets import Dataset

def load_and_prepare_dataset(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    
    # Convert to Hugging Face dataset
    dataset = Dataset.from_list(data)
    return dataset

# Load your dataset
dataset = load_and_prepare_dataset('your_dataset.jsonl')