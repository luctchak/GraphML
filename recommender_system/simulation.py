import numpy as np
import networkx as nx
from ALS import ALS
import pandas as pd
from suggest_one_film import *
from pick_random_user import pick_random_user
from build_films_graph import build_film_graph
from build_films_graph import build_film_clusters
from load_data import remove_duplicate
import math
import matplotlib.pyplot as plt
from load_data import load_data
from load_data import filter_data
import random


random.seed(11)
num_films_to_recommend = 30
minimum_number_of_films_rated = 50


def simulation(number_of_user_to_test, number_of_it_per_user):
    data = load_data("../data/u.data")
    data = filter_data(data)
    data = remove_duplicate(data)
    cumulated_reward_random = np.zeros(num_films_to_recommend)
    cumulated_reward_dist = np.zeros(num_films_to_recommend)
    cumulated_reward_kmeans = np.zeros(num_films_to_recommend)

    RMSE_random = np.zeros(num_films_to_recommend)
    RMSE_dist = np.zeros(num_films_to_recommend)
    RMSE_kmeans = np.zeros(num_films_to_recommend)


    d = 10
    num_cluster = 10
    it_max = 10

    # find the number of unique users
    num_users = len(list(set(data[:, 0])))
    # find the number of unique films
    num_items = len(list(set(data[:, 1])))
    # Select a set of random user for simulation
    random_users_selected = []
    nb_it = 0
    while len(random_users_selected) < number_of_user_to_test:
        random_user = pick_random_user(data, num_users, minimum_number_of_films_rated)
        nb_it += 1
        if not (random_user in random_users_selected):
            random_users_selected.append(random_user)
        if nb_it > 1000:
            print 'probably not enough users. Lower the minimum_number_of_films_rated'
            break

    # initialisation of als
    als = ALS(d, num_users, num_items, 'row', 'col', 'val', num_iters=10, verbose=True)

    for random_user_selected in random_users_selected:
        # Extract the list of films id for which we know the random user's ratings
        candidate_set = data[:, 1][data[:, 0] == random_user_selected]

        # remove the user randomly selected from the DB
        train = {}
        train['row'] = data[:, 0][data[:, 0] != random_user_selected]
        train['col'] = data[:, 1][data[:, 0] != random_user_selected]
        train['val'] = data[:, 2][data[:, 0] != random_user_selected]

        # Get the first decomposition R = U*V
        print "Fitting..."
        als.fit(train)
        print "Done."

        mem_train = als.train
        mem_U = als.U
        # Compute distance matrix of films based on V
        print "Building_film_graph..."
        distances = build_film_graph(als.V)
        print "Done."

        # Generate a graph from this distance matrix (G is fixed until the end)
        G = nx.from_numpy_matrix(distances)

        # Apply kmeans to get cluster assigment of films
        clusters_assignment = build_film_clusters(als.V, num_cluster)
        intermediate = data[data[:, 0] == random_user_selected]
        for recommendation_method in range(0,3):
            for it in range(0, number_of_it_per_user):
                als.train = mem_train
                als.U = mem_U
                # init values
                ever_seen = []
                R_user = np.zeros(num_items)



                for i in range(0, num_films_to_recommend):
                    #    recommendation = suggest_one_film(G, R_user, ever_seen, candidate_set)
                    #    if recommendation == -1:
                    #        print 'we explored all the possible solutions'
                    #        break
                    # uncomment bellow to use recommendation with kmeans
                    if recommendation_method == 0:
                        recommendation = suggest_one_film(G, R_user, ever_seen, candidate_set)
                    if recommendation_method == 1:
                        recommendation = suggest_one_film_kmeans(clusters_assignment, R_user, ever_seen, num_cluster, candidate_set, als.V)
                    if recommendation_method == 2:
                        recommendation = suggest_one_film_random(R_user, ever_seen, candidate_set, it_max)

                    if recommendation == -1:
                        print 'we explored all the possible solutions'
                        break

                    recommendation = int(recommendation)
                    reward = intermediate[intermediate[:, 1] == recommendation][0, 2]

                    if recommendation_method == 0:
                        cumulated_reward_dist[i] += reward
                        RMSE_dist[i] += RMSE(R_user, candidate_set, intermediate)
                    if recommendation_method == 1:
                        cumulated_reward_kmeans[i] += reward
                        RMSE_kmeans[i] += RMSE(R_user, candidate_set, intermediate)
                    if recommendation_method == 2:
                        cumulated_reward_random[i] += reward
                        RMSE_random[i] += RMSE(R_user, candidate_set, intermediate)

                    ever_seen.append(recommendation)
                    # add the value to train
                    als.train[random_user_selected, recommendation] = reward
                    # get the indices
                    indices = als.train[random_user_selected].nonzero()[1]
                    R_u = als.train[random_user_selected, indices]
                    als.U[random_user_selected, :] = als.update(indices, als.V, R_u.toarray().T)
                    # R_user[i] = np.dot(als.U[random_user_selected, :],als.V[i,:]) for all i \in {1,...,num_items}
                    R_user = np.einsum('ij,ij->i', np.tile(als.U[random_user_selected, :], [len(als.V), 1]), als.V)


    # makes it cumulative
    for i in range(1, num_films_to_recommend):
        cumulated_reward_random[i] += cumulated_reward_random[i-1]
        cumulated_reward_dist[i] += cumulated_reward_dist[i-1]
        cumulated_reward_kmeans[i] += cumulated_reward_kmeans[i-1]

    # normalize
    cumulated_reward_random /= number_of_it_per_user*number_of_user_to_test
    cumulated_reward_dist /= number_of_it_per_user*number_of_user_to_test
    cumulated_reward_kmeans /= number_of_it_per_user*number_of_user_to_test
    RMSE_dist /= number_of_it_per_user*number_of_user_to_test
    RMSE_kmeans /= number_of_it_per_user*number_of_user_to_test
    RMSE_random /= number_of_it_per_user*number_of_user_to_test

    #TODO : pourquoi ca affiche pas les labels

    plt.plot(range(0, 30), cumulated_reward_random, 'r', label='random')  # plotting t,a separately
    plt.plot(range(0, 30), cumulated_reward_dist, 'b', label='dist')    # plotting t,b separately
    plt.plot(range(0, 30), cumulated_reward_kmeans, 'g', label='k-means')  # plotting t,c separately

    plt.figure()
    plt.plot(range(0, 30), RMSE_random, 'r', label='random')  # plotting t,a separately
    plt.plot(range(0, 30), RMSE_dist, 'b', label='dist')    # plotting t,b separately
    plt.plot(range(0, 30), RMSE_kmeans, 'g', label='k-means')  # plotting t,c separately

    plt.show()

    return 0



def RMSE(R_user, candidate_set, intermediate):
    sum = 0
    for i in candidate_set:
        sum += (R_user[i] - intermediate[intermediate[:, 1] == i][0, 2])**2
    return sum/len(candidate_set)