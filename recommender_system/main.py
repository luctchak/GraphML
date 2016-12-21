import numpy as np
import Networkx as nx
from ALS import ALS
from suggest_one_film import suggest_one_film
from build_films_graph import build_film_graph

num_films_to_recommend = 100

def main():
    data = np.loadtxt("../data/u.data", dtype=int)
    d = 10
    # find the number of unique user_id elements
    num_users = len(list(set(data[:, 0])))
    num_items = len(list(set(data[:, 1])))
    user_col = data[:, 0]
    item_col = data[:, 1]
    rating_col = data[:, 2]
    als = ALS(d, num_users, num_items, user_col, item_col, rating_col)

    #TODO : See how to get R U and V from als

    # Compute distance matrix of films based on V
    distances = build_film_graph(V)

    # Generate a graph from this distance matrix (G is fixed until the end)
    G = nx.from_numpy_matrix(distances)


    ever_seen = []
    R_user = R[index, :]
    for i in range(0, num_films_to_recommend):
        recommendation = suggest_one_film(G, R_user, ever_seen)


main()


