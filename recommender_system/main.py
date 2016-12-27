import numpy as np
import networkx as nx
from ALS import ALS
from suggest_one_film import suggest_one_film
from build_films_graph import build_film_graph
from build_films_graph import build_film_clusters
from load_data import load_data
import random

num_films_to_recommend = 100

def main():
    data = load_data("../data/u.data")
    # data = filter(data, blablabla)

    #TODO : dimension correctly these parameter (maybe dynamically)
    d = 10
    num_cluster = 10

    # Select a random user for test
    random_user_selected = random.randint(0, len(list(set(data[:, 0]))))
    # find the number of unique users
    num_users = len(list(set(data[:, 0])))
    # find the number of unique films
    num_items = len(list(set(data[:, 1])))

    # remove the user randomly selected from the DB
    user_col = data[:, 0][data[:, 0] != random_user_selected]
    item_col = data[:, 1][data[:, 0] != random_user_selected]
    rating_col = data[:, 2][data[:, 0] != random_user_selected]

    # initialisation of als
    als = ALS(d, num_users, num_items, user_col, item_col, rating_col)

    #TODO : See how to get R U and V from als

    # Compute distance matrix of films based on V
    distances = build_film_graph(V)

    # Generate a graph from this distance matrix (G is fixed until the end)
    G = nx.from_numpy_matrix(distances)

    # Apply kmeans to get cluster assigment of films
    cluster_assigment = build_film_clusters(V, num_cluster)


    # init values
    ever_seen = []
    R_user = np.zeros((num_films_to_recommend, 1))
    cumulated_reward = np.zeros((num_films_to_recommend, 1))

    for i in range(0, num_films_to_recommend):
        recommendation = suggest_one_film(G, R_user, ever_seen)

        # uncomment bellow to use recommendation with kmeans
        # recommendation = suggest_one_film_kmeans(clusters_assigment, R_user, ever_seen, num_cluster)

        ever_seen = [ever_seen, recommendation]
        #TODO : find reward (not sure my code works) and update R_user
        reward = data[data[1, :] == random_user_selected][data[2, :] == recommendation][3]
        R_user = ...
main()


