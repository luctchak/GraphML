import numpy as np
from scipy import *
from scipy.sparse import *


# build the film_graph as a sparse matrix for a given input matrix
# V of size m x d where m is the number of film and d is the number of latent
# factors (i.e. the parameter chosen for decomposing R = UV)
MAX_DISTANCE = sys.float_info.max

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
            return beta-dist
    if distance_type is "inf":
        dist = np.max(abs(v_i-v_j))
        if dist < beta:
            return beta-dist

    return 0
