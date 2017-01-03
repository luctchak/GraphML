import numpy as np
from scipy import *
from sklearn.cluster import KMeans


# build the film_graph as a sparse matrix for a given input matrix
# V of size m x d where m is the number of film and d is the number of latent
# factors (i.e. the parameter chosen for decomposing R = UV)

def build_film_graph(V):
    num_film = len(V[:, 0])
    distances = np.zeros((num_film, num_film), dtype=float)
    for i in range(0, num_film):
        for j in range(0, num_film):
            distances[i, j] = distance(V[i, :], V[j, :], "L2", 1)

    return distances


def distance(v_i, v_j, distance_type, beta):
    if distance_type is "L2":
        dist = np.linalg.norm(v_i-v_j)
        if dist < beta:
            return dist
    if distance_type is "inf":
        dist = np.max(abs(v_i-v_j))
        if dist < beta:
            return dist

    return 0


# build a vector that assign each film to a cluster using kmeans
def build_film_clusters(V):
    is_not_after_gap = True
    inertias = np.zeros(10)
    gaps = np.zeros(9)
    # Naive method to estimate the number of cluster : taking the maximum gap
    for i in range(1, 10):
        inertias[i-1] = KMeans(n_clusters=i, random_state=0).fit(V).inertia_

    for i in range(0, 9):
        gaps[i] = inertias[i+1]-inertias[i]

    num_cluster = argmax(gaps) + 2
    print "num_cluster", num_cluster
    return KMeans(n_clusters=num_cluster, random_state=0).fit(V).labels_
