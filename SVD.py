import pickle
import math
import numpy as np
np.set_printoptions(threshold=np.inf)

movie_size = 2000           #INCLUSIVE OF 2000th movie
user_size = 1500            #INCLUSIVE OF 1500th movie

movie_pickle = open("movie_file.txt", 'rb')
rating_pickle = open("rating_file.txt", 'rb')

movie_dict = pickle.load(movie_pickle)
rating_dict  = pickle.load(rating_pickle)

movieIds = movie_dict.keys()
userIds = rating_dict.keys()

user_rating_matrix = [0] * (len(userIds) + 1)

for i in range(0, len(user_rating_matrix)):
    user_rating_matrix[i] = [0] * (movie_size+1)          #Possible Change

for user in userIds:
    user_movies = rating_dict[user].keys()
    for movie in user_movies:
        user_rating_matrix[int(user)][int(movie)] = float(rating_dict[user][movie])

user_array = np.array(user_rating_matrix)
# array_AAT = user_array*(user_array.T)
# eigenvalues, eigenvectors = np.linalg.eig(array_AAT)
# print(user_array.shape)
array_AAT = np.dot(user_array,(user_array.T))
eigenvalues, eigenvectors_AAT = np.linalg.eig(array_AAT)

rank = 0
for i in eigenvalues:
    if(i.real != 0):
        rank = rank + 1
# print(rank)

new_eigenvalues =[]
for i in range(0,len(eigenvalues)-1):
    new_eigenvalues.append((eigenvalues[i]).real)

original_size = len(new_eigenvalues)
for i in range(rank,original_size):
    del new_eigenvalues[rank]

# print(sigma)

#Since eigenvectors_AAT are placed in rows
U = eigenvectors_AAT

array_ATA = np.dot((user_array.T),user_array)
eigenvalues, eigenvectors_ATA = np.linalg.eig(array_ATA)

V = eigenvectors_ATA.T

# eigenvalues = new_eigenvalues
index = 1
sigma = []
sigma.insert(0,( [0]*(len(new_eigenvalues)+1) ))
for value in new_eigenvalues:
    p = [0] * (len(new_eigenvalues)+1)
    p[index] = value
    sigma.insert(index,p)
    index = index +1


# for i in range(rank,original_size):
#     np.delete(U, , 0)

# TODO U V and Sigma contain zeros must eliminate

# print(U.shape,len(sigma),V.shape)

# new_A = np.dot(np.dot(U,sigma),VT)


# To obtain values use these to print:
print(U)
# for i in U:
#     print(i)
# for i in V:
#     print(i)
# for i in sigma:
#     print(i)
# for i in new_eigenvalues:
#     print(i)
# for i in new_A:
#     print(i)
