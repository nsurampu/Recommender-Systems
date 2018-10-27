import pickle
import math
import numpy as np
np.set_printoptions(threshold=np.inf)

def energy(A):
    A = A*A
    return A.sum()

def RMSE(user_array,FinalA):
    error = user_array - FinalA
    # error = error[1:,:]
    sqerror = error*error
    print(sqerror.size)
    RMSE = sqerror.sum()/(sqerror.size)
    RMSE = math.sqrt(RMSE)
    print("RMSE is",RMSE)

def SVD(user_array):
    UAT = user_array.T
    array_AAT = np.dot(user_array, (UAT))
    eigenvalues, eigenvectors_AAT = np.linalg.eig(array_AAT)
    eigenvalues[eigenvalues < 1.0e-17] = 0
    #Finding rank
    rank = 0
    for i in eigenvalues:
        rank = rank + 1
        if(i.imag != 0):
            break
    rank = rank -1
    print("Rank: ",rank)
    # Rank = Row that contains last non zero value
    #Reducing size of eigenvalues to only include the actual rank
    eigenvalues = eigenvalues[0:(rank-1)]
    eigenvalues = eigenvalues.real
    #Build sigma
    sigma = np.diag(eigenvalues)
    #U of SVD with
    U = eigenvectors_AAT
    array_ATA = np.dot((UAT), user_array)
    eigenvalues_irr, eigenvectors_ATA = np.linalg.eig(array_ATA)
    V = eigenvectors_ATA
    # Slicing to match Size
    U = U[:, 0:rank-1]
    V = V[:, 0:rank-1]
    print("Size of U,V",U.shape, V.shape)
    return U,sigma,V,eigenvalues

movie_size = 2000  # INCLUSIVE OF 2000th movie
user_size = 1500  # INCLUSIVE OF 1500th movie

movie_pickle = open("movie_file.txt", 'rb')
rating_pickle = open("rating_file.txt", 'rb')

movie_dict = pickle.load(movie_pickle)
rating_dict = pickle.load(rating_pickle)

movieIds = movie_dict.keys()
userIds = rating_dict.keys()

user_rating_matrix = [0] * (len(userIds) + 1)

for i in range(0, len(user_rating_matrix)):
    user_rating_matrix[i] = [0] * (movie_size + 1)  # Possible Change

for user in userIds:
    user_movies = rating_dict[user].keys()
    for movie in user_movies:
        user_rating_matrix[int(user)][int(movie)] = float(
            rating_dict[user][movie])

user_array = np.array(user_rating_matrix)
user_array = user_array[1:,:]

U,sigma,V,eigenvalues = SVD(user_array)

VT = V.T
U = U.real
V = V.real
new_A = np.dot(U,sigma)
FinalA = np.dot(new_A,VT)

print(FinalA.shape)

energy = energy(eigenvalues)
# Reverse the eigenvalue np array
Reduction_array = np.empty([1])
for i in range(eigenvalue.size,0,-1):
    temp = eigenvalue[0:i]
    temp_energy = energy
    # if()


error = RMSE(user_array,FinalA)