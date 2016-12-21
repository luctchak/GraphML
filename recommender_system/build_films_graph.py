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
            distances[i, j] = distance(V[i, :], V[j, :], "L2", 0.1)

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
def build_film_clusters(V, num_classes):
    return KMeans(n_clusters=num_classes, random_state=0).fit(V)
