import numpy as np


# Loads the full DB
# To use it : data = load_data("../data/u.data")
# Output : np.array of size (100000,4)
def load_data(file_path):
    return np.loadtxt(file_path, dtype=int)


# In u.data you have 4 cols representing respectively [user id | item id | rating | timestamp]
# returns a sub-database of 444 films and 863 users
# when using data/u.data

def filter_data(all_data):
    selected_lines = np.array([[0, 0, 0, 0]])
    film_ids = np.unique(all_data[:, 1])
    for film_id in film_ids:
        rows_indices = np.where(all_data[:, 1] == film_id)[0]
        rows = all_data[rows_indices, :]
        mean_rating = rows[:, 2].mean()
        # print mean_rating
        if 2.5 > mean_rating > 2.2:
            selected_lines = np.concatenate((selected_lines, rows), axis=0)
    data_filtered = selected_lines[1:, :]

    # Map the unfiltered id of films and users to new consecutive ids between 0
    # and num_user_unfiltered/num_film_unfiltered
    film_mapper = np.zeros(len(all_data[:, 0])+1)-1
    user_mapper = np.zeros(len(all_data[:, 0])+1)-1
    cpt_film = 0
    cpt_user = 0
    for i in range(0, len(data_filtered[:, 0])):
        if user_mapper[data_filtered[i, 0]] != -1:
            data_filtered[i, 0] = user_mapper[data_filtered[i, 0]]
        else:
            data_filtered[i, 0] = cpt_user
            user_mapper[data_filtered[i, 0]] = cpt_user
            cpt_user += 1

        if film_mapper[data_filtered[i, 1]] != -1:
            data_filtered[i, 1] = film_mapper[data_filtered[i, 1]]
        else:
            data_filtered[i, 1] = cpt_film
            film_mapper[data_filtered[i, 1]] = cpt_film
            cpt_film += 1

    return data_filtered


