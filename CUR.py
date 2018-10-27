import pickle
import math
import numpy as np
import os
from cur import cur_decomposition
np.set_printoptions(threshold=np.inf)
np.random.seed(30)

'''
Takes the user rating matrix and returns the computed C, U and R matrices.
If it finds that the file CUR_Matrices.txt already exists, it unpickles it and returns them,
else it computes them, pickles them and then returns them, You can recompute these matrices 
using the boolean flag recomputeMatrix .

Parameters:
-----------
A : matrix that has to be decomposed
recomputeMatrix : as name suggests, recompute the decomposition matrices, pickle them and return the matrices 

returns:
--------
matrices C,U,R

'''

def CUR(A,num_dimensions,recomputeMatrix=False):
    
    file = 'CUR_Matrices.txt'

    if recomputeMatrix or not os.path.exists(file):
        
        num_rows = num_columns = num_dimensions
    
        # Computing C matrix
        temp = np.power(A,2)
        p_column = np.sum(temp,axis=0)/np.sum(temp)
        selected_columns = np.random.choice(A.shape[1],size=num_columns,p=p_column)
        # selected_columns = np.random.choice(A.shape[1],replace=False,size=num_columns,p=p_column)
        temp_C = A[:,selected_columns]
        column_scaling_factor = np.sqrt(p_column[selected_columns] * num_columns) 
        # print(temp_C.shape,len(column_scaling_factor))
        C = temp_C/column_scaling_factor
        # print('C computed')

        # Computing R matrix
        temp = np.power(A,2)
        p_rows = np.sum(temp,axis=1)/np.sum(temp)
        # selected_rows = np.random.choice(A.shape[0],replace=False,size=num_rows,p=p_rows)
        selected_rows = np.random.choice(A.shape[0],size=num_rows,p=p_rows)
        temp_R = A[selected_rows,:].T
        rows_scaling_factor = np.sqrt(p_rows[selected_rows] * num_rows)
        R = temp_R/rows_scaling_factor
        R = R.T
        # print('R ',R.shape)
        # print('R computed')

        # compute U
        W = A[selected_rows,:][:,selected_columns]
        # SVD for W
        W_WT = np.dot(W,W.T)
        WT_W = np.dot(W.T,W)
        # eigenvalue decomposition of W WT
        eigenvalues_W_WT, X = np.linalg.eig(W_WT)
        # print(eigenvalues_W_WT)
        idx = np.argsort(eigenvalues_W_WT)
        idx = idx[::-1]
        eigenvalues_W_WT = eigenvalues_W_WT[idx]
        eigenvalues_W_WT[np.abs(eigenvalues_W_WT) <= 1e-10] = 0
        X = X[:,idx]  
        X = X.real
        # eigenvalue decomposition of WT W
        eigenvalues_WT_W, Y = np.linalg.eig(WT_W)
        idx = np.argsort(eigenvalues_WT_W)
        idx = idx[::-1]
        eigenvalues_WT_W = eigenvalues_WT_W[idx]
        eigenvalues_WT_W[np.abs(eigenvalues_WT_W) <= 1e-10] = 1e200
        # print(eigenvalues_WT_W[eigenvalues_WT_W == 0])
        Y = Y[:,idx]
        Y = Y.real
        Z_plus = np.eye(eigenvalues_WT_W.shape[0])
        Z_plus = Z_plus*1/eigenvalues_WT_W
        Z_plus[Z_plus == 1e-200] = 0 
        U = np.dot(Y,Z_plus)
        U = np.dot(U,X.T)
        U = U.real
        # save file
        with open(file,'wb') as f:
            data = {}
            data['C'] = C
            data['R'] = R
            data['U'] = U            
            # save pickled data
            pickle.dump(data,f)
    else:
        with open(file,'rb') as f:
            data = pickle.load(f) 
            print('done')
            C = data['C']
            R = data['R']
            U = data['U']
            # print(C.shape,R.shape,U.shape) 

    return C,U,R
'''
Calculates the reconstruction error incurred after CUR decomposition

Parameters:
-----------
A: original matrix
C,U,R: matrices obtained after decomposition

returns:
--------
error: reconstruction error incurred while decomposing

'''
def reconstructionError(originalMatrix,C,U,R):
    reconstructedMatrix = np.dot(C,U)
    reconstructedMatrix = np.dot(reconstructedMatrix,R)

    error = np.sum(np.power((originalMatrix-reconstructedMatrix),2))
    error = np.power(error,0.5)
    return error

if __name__=='__main__':
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

    user_rating_matrix = np.array(user_rating_matrix)
    min_error = 1e50
    min_error_index = 100
    # for i in range(1,100):
    #     C,U,R = CUR(user_rating_matrix,605,recomputeMatrix=True)  # found experimentally.. assuming equal to the rank of the matrix
    #     error = reconstructionError(user_rating_matrix,C,U,R)
    #     if error < min_error:
    #         min_error = error
    #         min_error_index = i
    #     print(min_error,i)
        
    # print(min_error,min_error_index)
    C,U,R = CUR(user_rating_matrix,605,recomputeMatrix=True)
    error = reconstructionError(user_rating_matrix,C,U,R)
    new_A = np.dot(np.dot(C,U),R)
    print('\n\n',user_rating_matrix[:10,:10])
    print(new_A[:10,:10])
    print('\n\n')
    print('error my implementation ',error)
    size_of_U = 370   # found experimentally
    C,U,R = cur_decomposition(user_rating_matrix,size_of_U)
    error = reconstructionError(user_rating_matrix,C,U,R)
    print('error library ',error)
    # new_A = np.dot(np.dot(C,U),R)
    # print(user_rating_matrix[:10,:10])
    # print(new_A[:10,:10])