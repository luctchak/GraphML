import networkx as nx
import numpy as np
import random
# Suggestion with the distance constraint

def suggest_one_film(G, R_user, ever_seen, candidate_set):
    """
    Parameters
    ==========
    distances : float matrix
        distances between films - 0 means infinity (used as adjacency matrix to build graph)
    R_user : float vector
        R_user[i] is the predicted rating value for film with id i
    everSeen : list of int
        list of films already seen by the user
    candidate_set : list of int
        list of ids for which we know the groundtruth (ideally huge)
    """
    if len(ever_seen) == 0:
        return random.choice(candidate_set)

    is_not_acceptable = True
    R = 10
    # Sorting films by estimated preference
    indexes = sorted(range(len(R_user)), key=R_user.__getitem__)
    # Keep only the elements for which we know the ground truth
    new_indexes = np.zeros(len(candidate_set), dtype=int)
    cpt = 0
    for i in range(0, len(indexes)):
        if indexes[i] in candidate_set:
            new_indexes[cpt] = indexes[i]
            cpt += 1
    i = -1
    while is_not_acceptable:
        i += 1
        if i >= len(new_indexes):
            print 'We explored all the solution in ground truth'
            return -1
        if i > len(R_user):
            i = 0
            R /= 2
        is_not_acceptable = check_validity_of_film(G, ever_seen, new_indexes[i], R)
    return new_indexes[i]

# Returns False is the film chosen is too close to an existing one


def check_validity_of_film(G, ever_seen, index, R):
    if index in ever_seen:
        return True

    for l in ever_seen[1:(len(ever_seen)-1)]:
        try:
            if nx.shortest_path(G, source=index, target=l) < R/(len(ever_seen)+1):
                print nx.shortest_path(G, source=index, target=l)
                return True
        except nx.NetworkXError:
            return True

    return False


def suggest_one_film_kmeans(clusters_assignment, R_user, ever_seen, num_cluster, candidate_set, V):
    # Case : we already explored all the clusters
    if len(ever_seen) > num_cluster:
        indexes = sorted(range(len(R_user)), key=R_user.__getitem__)
        # Keep only the elements for which we know the ground truth
        new_indexes = np.zeros(len(candidate_set),dtype=int)
        cpt = 0
        for i in range(0, len(indexes)):
            if indexes[i] in candidate_set:
                new_indexes[cpt] = indexes[i]
                cpt += 1
        i = 0
        while new_indexes[i] in ever_seen:
            if i >= len(new_indexes):
                print 'We explored all the solution in ground truth'
                return -1
            i += 1

        return new_indexes[i]
    else:
        # We take the best predicted film from the cluster len(ever_seen)
        sub_R_user = np.copy(R_user)
        for i in range(0, len(sub_R_user)):
            #TODO : lever le warning engendre par cette ligne
            if clusters_assignment.predict(V[i,:])!=len(ever_seen):
            # if clusters_assignment.labels_[i] != len(ever_seen):
                sub_R_user[i] = -1000  # should be enough

        argmaxs = np.argwhere(sub_R_user == np.amax(sub_R_user))
        items_pool = [i for i in argmaxs if i in candidate_set]
        if len(items_pool) == 0:
            print "Kmeans suggestion: empty intersection, randomly suggesting"
            items_pool = [i for i in candidate_set if i not in ever_seen]
            return random.choice(items_pool)

        return items_pool[0]


def suggest_one_film_random(R_user, ever_seen, candidate_set, it_max):
    if it_max > len(ever_seen):
        random_suggestion = candidate_set[random.randint(0, len(list(set(candidate_set)))-1)]
        while random_suggestion in ever_seen:
            random_suggestion = candidate_set[random.randint(0, len(list(set(candidate_set)))-1)]
        return random_suggestion
    else:
        indexes = sorted(range(len(R_user)), key=R_user.__getitem__)
        # Keep only the elements for which we know the ground truth
        new_indexes = np.zeros(len(candidate_set), dtype=int)
        cpt = 0
        for i in range(0, len(indexes)):
            if indexes[i] in candidate_set:
                new_indexes[cpt] = indexes[i]
                cpt += 1
        i = 0
        while new_indexes[i] in ever_seen:
            if i >= len(new_indexes):
                print 'We explored all the solution in ground truth'
                return -1
            i += 1

        return new_indexes[i]