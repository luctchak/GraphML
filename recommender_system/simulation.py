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

num_films_to_recommend = 40
minimum_number_of_films_rated = 60


def simulation(number_of_user_to_test, number_of_it_per_user):
    data = load_data("../data/u.data")
    data = remove_duplicate(data)
    data = filter_data(data)

    cumulated_reward_random = np.zeros(num_films_to_recommend)
    cumulated_reward_dist = np.zeros(num_films_to_recommend)
    cumulated_reward_kmeans = np.zeros(num_films_to_recommend)

    RMSE_random = np.zeros(num_films_to_recommend)
    RMSE_dist = np.zeros(num_films_to_recommend)
    RMSE_kmeans = np.zeros(num_films_to_recommend)

    d = 4
    it_max = 10
    # find the number of unique users
    num_users = len(np.unique(data[:, 0]))
    # find the number of unique films
    num_items = len(np.unique(data[:, 1]))
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
    print "number_of_films_in_DB", num_items
    print "number_of_users_in_DB", num_users
    print "number_of_ratings    ", len(data[:, 0])
    print "random_users_selected", random_users_selected
    # initialisation of als
    als = ALS(d, num_users, num_items, "Users", "Movies", "Ratings", num_iters=20, verbose=True,lbda = 0.1,lbda2 = 0.1)

    for random_user_selected in random_users_selected:

        # Extract the list of films id for which we know the random user's ratings
        candidate_set = data[:, 1][data[:, 0] == random_user_selected]
        print "candidate_set", candidate_set
        # remove the user randomly selected from the DB
        R_dict = dict()
        R_dict["Users"] = data[:, 0][data[:, 0] != random_user_selected]
        R_dict["Movies"] = data[:, 1][data[:, 0] != random_user_selected]
        R_dict["Ratings"] = data[:, 2][data[:, 0] != random_user_selected]
        mean = data[:, 2][data[:, 0] != random_user_selected].mean()

        # Get the first decomposition R = U*V
        print "Fitting..."
        als.fit(R_dict)
        print "Done."
        print "Global RMSE", RMSE_total(als.U.dot(als.V.T), data)


        mem_train = als.train.copy()
        mem_U = als.U.copy()

        # Compute distance matrix of films based on V
        print "Building_film_graph..."
        distances = build_film_graph(als.V)
        print "Done."


        # Generate a graph from this distance matrix (G is fixed until the end)
        G = nx.from_numpy_matrix(distances)

        # Apply kmeans to get cluster assigment of films
        clusters_assignment = build_film_clusters(als.V)
        intermediate = data[data[:, 0] == random_user_selected]


        for recommendation_method in range(0, 3):
            for it in range(0, number_of_it_per_user):
                als.train = mem_train.copy()
                als.U = mem_U.copy()
                # init values
                ever_seen = []
                R_user = np.zeros(num_items)
                for i in range(0, num_films_to_recommend):
                    #    recommendation = suggest_one_film(G, R_user, ever_seen, candidate_set)
                    #    if recommendation == -1:
                    #        print 'we explored all the possible solutions'
                    #        break
                    # uncomment bellow to use recommendation with kmeans
                    if recommendation_method == 2:
                        recommendation = suggest_one_film(G, R_user, ever_seen, candidate_set)
                    if recommendation_method == 1:
                        recommendation = suggest_one_film_kmeans(clusters_assignment, R_user, ever_seen, candidate_set)
                    if recommendation_method == 0:
                        recommendation = suggest_one_film_random(R_user, ever_seen, candidate_set, it_max)

                    if recommendation == -1:
                        print 'we explored all the possible solutions'
                        break

                    recommendation = int(recommendation)
                    reward = intermediate[intermediate[:, 1] == recommendation][0, 2]
                    verbose = False


                    # update the RMSE

                    if recommendation_method == 2:
                        cumulated_reward_dist[i] += reward
                        RMSE_dist[i] += RMSE(R_user, candidate_set, intermediate, mean, verbose)
                    if recommendation_method == 1:
                        cumulated_reward_kmeans[i] += reward
                        RMSE_kmeans[i] += RMSE(R_user, candidate_set, intermediate, mean, verbose)
                    if recommendation_method == 0:
                        cumulated_reward_random[i] += reward
                        RMSE_random[i] += RMSE(R_user, candidate_set, intermediate, mean, verbose)

                    ever_seen.append(recommendation)
                    # add the value to train
                    als.train[random_user_selected, recommendation] = reward
                    # get the indices
                    indices = als.train[random_user_selected].nonzero()[1]

                    # update R_u
                    R_u = als.train[random_user_selected, indices]
                    Hix = als.V[indices, :]
                    HH = Hix.T.dot(Hix)
                    M = HH + np.diag(als.lbda*len(R_u.toarray().T)*np.ones(als.d))
                    als.U[random_user_selected, :] = np.linalg.solve(M, Hix.T.dot(R_u.toarray().T)).reshape(als.d)
                    for i in candidate_set:
                        R_user[i] = als.U[random_user_selected, :].dot(als.V[i, :].T)

                if verbose:
                    print '\n'
                    print '='*40
                    print "recommendation_method", recommendation_method
                    print "it", it
                    print "R_user[ever_seen]", R_user[ever_seen]
                    print '='*40

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

    print "RMSE_dist", RMSE_dist
    print "RMSE_kmeans", RMSE_kmeans
    print "RMSE_random", RMSE_random

    #plt.ion()
    plt.plot(range(0, num_films_to_recommend), cumulated_reward_random, 'r', label='random')  # plotting t,a separately
    plt.plot(range(0, num_films_to_recommend), cumulated_reward_dist, 'b', label='dist')    # plotting t,b separately
    plt.plot(range(0, num_films_to_recommend), cumulated_reward_kmeans, 'g', label='k-means')  # plotting t,c separately
    plt.legend(loc=0)
    plt.title('cumulative reward')
    plt.figure()
    plt.plot(range(1, num_films_to_recommend), RMSE_random[1: num_films_to_recommend], 'r', label='random')  # plotting t,a separately
    plt.plot(range(1, num_films_to_recommend), RMSE_dist[1: num_films_to_recommend], 'b', label='dist')    # plotting t,b separately
    plt.plot(range(1, num_films_to_recommend), RMSE_kmeans[1: num_films_to_recommend], 'g', label='k-means')  # plotting t,c separately
    plt.legend(loc=0)
    plt.title('RMSE')
    plt.show()

    return 0



def RMSE(R_user, candidate_set, intermediate, verbose = False,  center = True):
    sum = 0
    if len(candidate_set) == 0:
        print "should not happen"
        return 0
    for i in candidate_set:
        if verbose:
            print "R_user[",i ,"]", R_user[i]
            print "ground_truth", intermediate[intermediate[:, 1] == i][0, 2]
        sum += (R_user[i] - intermediate[intermediate[:, 1] == i][0, 2])**2
    return (sum/len(candidate_set))**0.5



def RMSE_total(train, data):
    sum = 0
    for i in range(len(data[:, 0])):
        sum += (train[data[i,0],data[i,1]]-data[i,2])**2

    return (sum/len(data))**0.5




