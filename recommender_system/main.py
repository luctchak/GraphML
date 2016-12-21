import numpy as np
import ALS


def main():
    data = np.loadtxt("../data/u.data", dtype=int)
    d = 10
    # find the number of unique user_id elements
    num_users = len(list(set(data[:, 1])))
    num_items = len(list(set(data[:, 2])))
    user_col = data[:, 1]
    item_col = data[:, 2]
    rating_col = data[:, 3]
    als = ALS(d, num_users, num_items, user_col, item_col, rating_col)
    print range(1,3)


