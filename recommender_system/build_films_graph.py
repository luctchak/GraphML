import numpy as np
from scipy import *
from scipy.sparse import *
# build the film_graph as a sparse matrix for a given input matrix
# V of size m x d where m is the number of film and d is the number of latent
# factors (i.e. the parameter chosen for decomposing R = UV)
def build_film_graph(V):
    num_film = len(V[:, 1])
    similarities = csr_matrix((num_film, num_film), dtype=float)
    for i in range(1,num_film):
        for j in range(1,num_film):
           similarities[i,j]= similarity(V[i,:], V[j,:], "L2", 0.1)



def similarity(v_i, v_j, distance_type, beta):
    if distance_type is "L2":
        dist = np.linalg.norm(v_i-v_j)
        if dist<beta:
            return beta-dist

    if distance_type is "inf":
        dist = np.max(abs(v_i-v_j))
        if dist<beta:
            return beta-dist

    return 0