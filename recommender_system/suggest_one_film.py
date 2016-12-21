import networkx as nx
import numpy as np
import random
# Suggestion with the distance constraint

def suggest_one_film(G, R_user, ever_seen):
    """
    Parameters
    ==========
    distances : float matrix
        distances between films - 0 means infinity (used as adjacency matrix to build graph)
    R_user : float vector
        R_user[i] is the predicted rating value for film with id i
    everSeen : list of int
        list of films already seen by the user
    """
    is_not_acceptable = True
    R = 10
    # Sorting films by estimated preference
    indexes = sorted(range(len(R_user)), key=R_user.__getitem__)
    i = -1
    while is_not_acceptable:
        i += 1
        if i > len(R_user):
            i = 0
            R /= 2
        is_not_acceptable = check_validity_of_film(G, ever_seen, indexes[i], R)

    return indexes[i]

# Returns False is the film chosen is too close to an existing one


def check_validity_of_film(G, ever_seen, index, R):
    for l in ever_seen:
        if nx.shortest_path(G, source=index, target=ever_seen[l]) < R/(len(ever_seen)+1):
            return True

    return False


def suggest_one_film_kmeans(clusters_assigment, R_user, ever_seen, num_cluster):
    # Case : we already explored all the clusters
    if len(ever_seen) > num_cluster:
        indexes = sorted(range(len(R_user)), key=R_user.__getitem__)
        i = 0
        while indexes[i] in ever_seen:
            i += 1
        return indexes[i]
    else:
        # We take the best predicted film from the cluster i
        sub_R_user = np.copy(R_user)
        for i in range(0, len(sub_R_user)):
            if clusters_assigment[i] != len(ever_seen):
                sub_R_user[i] = -1000  # should be enough
        return np.argmax(sub_R_user)


def suggest_one_film_random(R_user, ever_seen):
    # Case : we already explored all the clusters
    random_suggestion = random.randint(0, len(list(set(R_user))))
    while random_suggestion in ever_seen:
        random_suggestion = random.randint(0, len(list(set(R_user))))
    return random_suggestion
