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

num_films_to_recommend = 50
minimum_number_of_films_rated = 50


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

    d = 20
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
    print "number_of_films_in_DB", num_items
    print "number_of_users_in_DB", num_users
    print "number_of_ratings    ", len(data[:, 0])
    print "random_users_selected", random_users_selected
    # initialisation of als
    als = ALS(d, num_users, num_items, 'row', 'col', 'val', num_iters=20, verbose=True)

    for random_user_selected in random_users_selected:
        #TODO : delete the following line
        random_user_selected = 69
        # Extract the list of films id for which we know the random user's ratings
        candidate_set = data[:, 1][data[:, 0] == random_user_selected]
        print "candidate_set", candidate_set
        # remove the user randomly selected from the DB

        train = dict()
        train['row'] = data[:, 0][data[:, 0] != random_user_selected]
        train['col'] = data[:, 1][data[:, 0] != random_user_selected]
        train['val'] = data[:, 2][data[:, 0] != random_user_selected]

        # Get the first decomposition R = U*V
        print "Fitting..."
        als.fit(train)
        print "Done."

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
                    #if i == 1:
                        # verbose = True
                    if recommendation_method == 2:
                        cumulated_reward_dist[i] += reward
                        RMSE_dist[i] += RMSE(R_user, candidate_set, intermediate, verbose)
                    if recommendation_method == 1:
                        cumulated_reward_kmeans[i] += reward
                        RMSE_kmeans[i] += RMSE(R_user, candidate_set, intermediate, verbose)
                    if recommendation_method == 0:
                        cumulated_reward_random[i] += reward
                        RMSE_random[i] += RMSE(R_user, candidate_set, intermediate, verbose)

                    ever_seen.append(recommendation)
                    # add the value to train
                    als.train[random_user_selected, recommendation] = reward
                    # get the indices
                    indices = als.train[random_user_selected].nonzero()[1]

                    R_u = als.train[random_user_selected, indices]



                    Hix = als.V[indices,:]
                    HH = Hix.T.dot(Hix)
                    #M = HH + np.diag(als.lbda*len(R_u.toarray().T)*np.ones(als.d))
                    M = HH + np.diag(1000*len(R_u.toarray().T)*np.ones(als.d))
                    copy = als.U[random_user_selected, :].copy()
                    als.U[random_user_selected, :] = np.linalg.solve(M,Hix.T.dot(R_u.toarray().T)).reshape(als.d)
                    print "delta U normalized", copy*als.U[random_user_selected, :].mean()/copy.mean() - als.U[random_user_selected, :]
                    #als.U[random_user_selected, :] = als.update(indices, als.V, R_u.toarray().T)
                    for i in candidate_set:
                        R_user[i] = np.dot(als.U[random_user_selected, :], als.V[i, :].T)

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
    plt.plot(range(0, num_films_to_recommend), RMSE_random, 'r', label='random')  # plotting t,a separately
    plt.plot(range(0, num_films_to_recommend), RMSE_dist, 'b', label='dist')    # plotting t,b separately
    plt.plot(range(0, num_films_to_recommend), RMSE_kmeans, 'g', label='k-means')  # plotting t,c separately
    plt.legend(loc=0)
    plt.title('RMSE')
    plt.show()

    return 0



def RMSE(R_user_to_copy, candidate_set, intermediate, verbose = False, center = True):
    sum = 0
    R_user = R_user_to_copy.copy()
    mean1 = R_user[candidate_set].mean()
    if center and mean1!=0:
        mean2 = 0
        for i in candidate_set:
            mean2 += intermediate[intermediate[:, 1] == i][0, 2]
        mean2/=len(candidate_set)
        R_user *= mean2/mean1

    if len(candidate_set) == 0:
        print "should not happen"
        return 0
    for i in candidate_set:
        if verbose:
            print "R_user[",i,"]", R_user[i]
            print "ground_truth", intermediate[intermediate[:, 1] == i][0, 2]
        sum += (R_user[i] - intermediate[intermediate[:, 1] == i][0, 2])**2
    return sum/len(candidate_set)




