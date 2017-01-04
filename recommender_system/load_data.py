import numpy as np


# Loads the full DB
# To use it : data = load_data("../data/u.data")
# Output : np.array of size (100000,4)
def load_data(file_path):
    return np.loadtxt(file_path, dtype=int)


# In u.data you have 4 cols representing respectively [user id | item id | rating | timestamp]

# Filter in order to only keep films rated averagely

def filter_data(all_data):
    selected_lines = np.array([[0, 0, 0]])
    film_ids = np.unique(all_data[:, 1])
    for film_id in film_ids:
        if len(all_data[:, 0][all_data[:, 1] == film_id]) > 50:
            selected_lines = np.concatenate((selected_lines, all_data[:, :][all_data[:, 1] == film_id]), axis=0)
    data_filtered = selected_lines[1:, :]

    # Map the unfiltered id of films and users to new consecutive ids between 0
    # and num_user_unfiltered/num_film_unfiltered
    ever_seen_film = []
    ever_seen_user = []

    for i in range(0, len(data_filtered[:, 0])):
        if data_filtered[i, 0] in ever_seen_user:
            data_filtered[i, 0] = ever_seen_user.index(data_filtered[i, 0])
        else:
            ever_seen_user.append(data_filtered[i, 0])
            data_filtered[i, 0] = ever_seen_user.index(data_filtered[i, 0])

        if data_filtered[i, 1] in ever_seen_film:
            data_filtered[i, 1] = ever_seen_film.index(data_filtered[i, 1])
        else:
            ever_seen_film.append(data_filtered[i, 1])
            data_filtered[i, 1] = ever_seen_film.index(data_filtered[i, 1])

    return data_filtered


# Remove duplicated lines after having removing the 4th col corresponding to the time
def remove_duplicate(all_data):
    all_data = all_data[:, 0:3]
    return unique_rows(all_data)



def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))