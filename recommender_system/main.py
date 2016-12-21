import numpy as np

def main():
    data = np.loadtxt("../data/u.data", dtype=int)
    d = 10
    num_user = data
    als = ALS()

