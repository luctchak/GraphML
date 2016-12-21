import Networkx as nx


def suggest_one_film(G, R_user, ever_seen):
    """
    Parameters
    ==========
    distances : float matrix
        similarities between films
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




