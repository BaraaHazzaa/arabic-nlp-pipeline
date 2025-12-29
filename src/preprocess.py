#G O A L S
# 1. Pull Data From Data.py
# 2. Preprocess The Data
# 3. Return The Preprocessed Data
from data import load_data

def preprocess_data(data):
    preprocessed_data=[]
    for item in data:
        # Example preprocessing: lowercasing text fields
        if 'text' in item:
            item['text'] = item['text'].lower()
        preprocessed_data.append(item)
    return preprocessed_data


def return_preprocessed_data(preprocessed_data):
    pass