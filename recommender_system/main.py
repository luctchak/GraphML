import numpy as np
import networkx as nx
from ALS import ALS
from suggest_one_film import suggest_one_film
from pick_random_user import pick_random_user
from build_films_graph import build_film_graph
from build_films_graph import build_film_clusters
from load_data import load_data
from load_data import filter_data
import random

num_films_to_recommend = 10
minimum_number_of_films_rated = 20

def main():
    data = load_data("../data/u.data")
    data = filter_data(data)

    #TODO : dimension correctly these parameters (maybe dynamically)
    d = 10
    num_cluster = 10

    # find the number of unique users
    num_users = len(list(set(data[:, 0])))
    # find the number of unique films
    num_items = len(list(set(data[:, 1])))
    # Select a random user for test
    random_user_selected = pick_random_user(data, num_users, minimum_number_of_films_rated)
    # Extract the list of films id for which we know the random user's ratings
    candidate_set = data[:, 1][data[:, 0] == random_user_selected]

    # initialisation of als
    als = ALS(d, num_users, num_items, 'row','col','val')

    # remove the user randomly selected from the DB
    train = {}
    train['row'] = data[:, 0][data[:, 0] != random_user_selected]
    train['col'] = data[:, 1][data[:, 0] != random_user_selected]
    train['val'] = data[:, 2][data[:, 0] != random_user_selected]

    # Get the first decomposition R = U*V
    print "Fitting..."
    als.fit(train)
    print "Done."


    # Compute distance matrix of films based on V
    print "Building_film_graph..."
    distances = build_film_graph(als.V)
    print "Done."

    # Generate a graph from this distance matrix (G is fixed until the end)
    G = nx.from_numpy_matrix(distances)

    # Apply kmeans to get cluster assigment of films
    cluster_assigment = build_film_clusters(als.V, num_cluster)


    # init values
    ever_seen = []
    rewards = []
    R_user = np.zeros(num_items)
    cumulated_reward = np.zeros((num_films_to_recommend, 1))

    for i in range(0, num_films_to_recommend):
        recommendation = suggest_one_film(G, R_user, ever_seen, candidate_set)
        if recommendation == -1:
            print 'we explored all the possible solutions'
            break
        # uncomment bellow to use recommendation with kmeans
        # recommendation = suggest_one_film_kmeans(clusters_assigment, R_user, ever_seen, num_cluster, candidate_set)
        print recommendation
        ever_seen = [ever_seen, recommendation]
        intermediate = data[data[:, 0] == random_user_selected]
        print intermediate[intermediate[:, 1] == recommendation][:, 2]
        reward = intermediate[intermediate[:, 1] == recommendation][:, 2]
        rewards = [rewards, reward]

        # R_user = . . .

        # add the value to train
        als.train[random_user_selected, recommendation] = reward
        # get the indices
        indices = als.train[random_user_selected].nonzero()[1]
        R_u = als.train[random_user_selected, indices]
        als.U[random_user_selected, :] = als.update(indices, als.V, R_u.toarray().T)

    print ever_seen
    print rewards

main()


