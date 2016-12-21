import numpy as np


def load_data(file_path):
    return np.loadtxt(file_path, dtype=int)

# To use it :
# data = load_data("../data/u.data")


