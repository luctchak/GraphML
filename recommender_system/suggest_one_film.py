def suggest_one_film(distances, R_user, everSeen):
    """
    Parameters
    ==========
    similarities : float matrix
        similarities between films
    R_user : float vector
        R_user[i] is the predicted rating value for film with id i
    everSeen : list of int
        list of films already seen by the user
    """
    is_acceptable = False
    while is_acceptable==False :
        is_acceptable = check_validity_of_film()

    return film_id_chosen