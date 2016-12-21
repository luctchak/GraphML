import numpy as np

# Load the full DB
# To use it : data = load_data("../data/u.data")

def load_data(file_path):
    return np.loadtxt(file_path, dtype=int)


# In u.data you have 4 cols representing respectively [user id | item id | rating | timestamp]


# Filter database for tests
def filter(data, eventual_extra_parameters)




    return data_filtered