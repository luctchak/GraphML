import numpy as np


def load_data(file_path):
	""" 
	Loads the full DB
	To use it : data = load_data("../data/u.data")
	Output : np.array of size (100000,4)
	"""
	return np.loadtxt(file_path, dtype=int)


# In u.data you have 4 cols representing respectively [user id | item id | rating | timestamp]

def filter(all_data):
	""" 
	returns a sub-database of 444 films and 863 users
	when using data/u.data 
	"""
	selected_lines = np.array([[0,0,0,0]])
	film_ids = np.unique(all_data[:,1])
	for film_id in film_ids: 
		rows_indices = np.where(all_data[:,1]==film_id)[0]
		rows = all_data[rows_indices,:]
		mean_rating = rows[:,2].mean()
		#print mean_rating
		if mean_rating > 2 and mean_rating < 3: 
			selected_lines = np.concatenate((selected_lines,rows), axis=0)
	return selected_lines[1:,:]


			
	

