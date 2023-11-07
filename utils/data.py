import numpy as np
import re
import matplotlib.pyplot as plt

def extract(file_path):
    df = []
    with open(file_path, 'r') as file:
        for row in file:
            col = re.split(r'\s+', row.strip())
            df.append(col)

    return np.array(df).astype(float)

def normalize_rad(r):
    if r > np.pi:
        r -= 2 * np.pi
    elif r < -np.pi:
        r += 2 * np.pi
    return r

def show_comparasion(t, real, pred):
    plt.scatter(t, real, marker='o', color='red', label='real')
    plt.scatter(t, pred, marker='o', color='blue', label='pred')
    plt.legend()
    plt.show()