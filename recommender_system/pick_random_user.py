import numpy as np

# pick one user who rated a lot of films

def pick_random_user(data, num_users, minimum_number_of_films_rated):
    random_permut = np.random.permutation(range(0, num_users))
    for i in random_permut:
        if len(data[:, 0][data[:, 0] == i] >= minimum_number_of_films_rated):
            return i
    print 'No such element exist, reduce the number of film rated'
    return -1
