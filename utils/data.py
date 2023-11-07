import numpy as np
import re

def extract(file_path):
    df = []
    with open(file_path, 'r') as file:
        for row in file:
            col = re.split(r'\s+', row.strip())
            df.append(col)

    return np.array(df).astype(float)