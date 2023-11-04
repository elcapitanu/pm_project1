import numpy as np
import re

file_path = "./data/data1.txt"

df = []

with open(file_path, 'r') as file:
    for row in file:
        col = re.split(r'\s+', row.strip())
        df.append(col)

data = np.array(df)