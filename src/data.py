#G O A L S
# 1. DEFINE THE DATA DIRECTORY
# 2. LOAD THE DATA FROM A FILE
# 3. RETURN THE DATA AS A LIST OF DICTIONARIES

import os
from config import DATA_DIR

def get_data_directory():
    return DATA_DIR


def load_data(file_name):
    file_path=os.path.join(get_data_directory(),file_name)
    data=[]
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Assuming each line is a JSON object
            data.append(eval(line.strip()))  # Using eval for simplicity; consider using json.loads for safety
    return data
