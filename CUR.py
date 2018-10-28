import pickle
import math
import numpy as np
import os
from cur import cur_decomposition
import time
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

def CUR(A,num_dimensions,recomputeMatrix=False,energy_needed=1.0):
    
    if energy_needed > 1:
        raise Exception('energy_needed should not exceed 1. The value of energy_needed was: {}'.format(energy_needed))

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
        print('C computed')

        # Computing R matrix
        temp = np.power(A,2)
        p_rows = np.sum(temp,axis=1)/np.sum(temp)
        # selected_rows = np.random.choice(A.shape[0],replace=False,size=num_rows,p=p_rows)
        selected_rows = np.random.choice(A.shape[0],size=num_rows,p=p_rows)
        temp_R = A[selected_rows,:].T
        rows_scaling_factor = np.sqrt(p_rows[selected_rows] * num_rows)
        R = temp_R/rows_scaling_factor
        R = R.T
        print('R computed')

        # compute U
        W = A[selected_rows,:][:,selected_columns]
        # SVD for W
        W_WT = np.dot(W,W.T)
        WT_W = np.dot(W.T,W)
        print('shapes',W_WT.shape,WT_W.shape)
        # eigenvalue decomposition of W WT
        eigenvalues_W_WT, X = np.linalg.eig(W_WT)
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
        Y = Y[:,idx]
        Y = Y.real

        # energy based selection, if necessary that is...
        if energy_needed != 1:        
            variances =  np.power(eigenvalues_W_WT,2)
            variances = variances.real
            total_energy = np.sum(variances)
            total_energy = total_energy.real
            # print('total energy ',total_energy)
            # print(variances[:10])
            
            index_to_slice = 0

            for i in range(variances.shape[0]):
                current_energy = np.sum(variances[:i])
                if current_energy >= energy_needed*total_energy:
                    index_to_slice = i
                else:
                    break
            eigenvalues_WT_W = eigenvalues_WT_W[:index_to_slice + 1]
            X = X[:,:index_to_slice + 1]
            Y = Y[:,:index_to_slice + 1]

        Z_plus = np.eye(eigenvalues_WT_W.shape[0])
        Z_plus = Z_plus*1/eigenvalues_WT_W
        Z_plus[Z_plus == 1e-200] = 0 
        U = np.dot(Y,Z_plus)
        U = np.dot(U,X.T)
        U = U.real
        eigenvalues_WT_W[np.abs(eigenvalues_WT_W) == 1e200] = 0
        # save file
        with open(file,'wb') as f:
            data = {}
            data['C'] = C
            data['R'] = R
            data['U'] = U     
            data['eigenvalues'] = eigenvalues_WT_W       
            # save pickled data
            pickle.dump(data,f)
    else:
        with open(file,'rb') as f:
            data = pickle.load(f) 
            print('done')
            C = data['C']
            R = data['R']
            U = data['U']
            eigenvalues_WT_W = data['eigenvalues']
            # print(C.shape,R.shape,U.shape)

        '''...check this...'''
    # if energy_needed != 1:
    #     energy = Energy(eigenvalues_WT_W)
    #     print('energy ',energy)
    #     # Reverse the eigenvalue np array
    #     Reduction_array = np.empty([1])
    #     for i in range(eigenvalues_WT_W.size,0,-1):
    #         temp = eigenvalues_WT_W[0:i]
    #         temp_Energy = Energy(temp)
    #         if(temp_Energy >= energy_needed * energy):
    #             Reduction_array = temp
    #             print(i)
    #         else:
    #             break
    #     size = Reduction_array.size
    #     print('lite ',size)
    #     print('zeros at',np.where(eigenvalues_WT_W == 0)[0])
    #     print(C.shape,U.shape,R.shape)
    #     Reduction_array = Reduction_array[0:(size-1)]
    #     Reduction_array = Reduction_array.real
    #     sigma_reduced = np.diag(Reduction_array)
    #     C_reduced = C[:,0:(size-1)]
    #     R_reduced = R[:,0:(size-1)]
    #     RT_reduced = R_reduced.T
    #     print('hehe ' ,C_reduced.shape,R_reduced.shape,sigma_reduced.shape)
    #     new_A_reduced = np.dot(C_reduced,sigma_reduced)
    #     ReducedA = np.dot(new_A_reduced,RT_reduced)
    #     print(ReducedA.shape)
    #     print(user_rating_matrix.shape)
        
    return C,U,R,eigenvalues_WT_W


'''
Calculates the Root mean Squared Error(RMSE) incurred after CUR decomposition

Parameters:
-----------
A: original matrix
C,U,R: matrices obtained after decomposition

returns:
--------
error: reconstruction error incurred while decomposing

'''


def rmse(originalMatrix,C,U,R):
    reconstructedMatrix = np.dot(C,U)
    reconstructedMatrix = np.dot(reconstructedMatrix,R)

    error = np.sum(np.power((originalMatrix-reconstructedMatrix),2))/(reconstructedMatrix.shape[0] * reconstructedMatrix.shape[1])
    error = np.power(error,0.5)
    return error

def query(R,q):
    start_time = time.clock()
    temp = np.dot(q,R)
    final = np.dot(temp,R.T)
    duration = time.clock() - start_time
    return final,duration

def precisionTopK(k,q,R):
    final = query(q,R)
    print(final.shape)
    final[final < 3.5] = 0
    final[final > 3.5] = 1
    q[q < 3.5] = 0
    q[q > 3.5] = 1
    idx = final.argsort()[::-1]
    final = final[idx]
    q = q[:,idx]
    prec_val = 0
    for i in range(0,k-1):
        if(final[i,0] == 1 and q[i,0] == 1):
            prec_val +=1
    prec_val = prec_val / k
    return prec_val

''' 

'''
def spearmanCoefficient(predicted_rating,test_rating):
    predicted_rank = np.argsort(predicted_rating)
    test_rank = np.argsort(test_rating)
    d = test_rank - predicted_rank
    d_squared = np.power(d,2)
    sum_d_squared = np.sum(d_squared)
    n = d.shape[0]
    rho = 1 - (6*sum_d_squared)/(n*(n**2 - 1))
    return rho




def Energy(A):
    A = A*A
    return A.sum()    

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

    # calculating best size (number of columns for fitting the data)
    
    # min_error = 1e50
    # min_error_index = 0
    # for i in range(1,user_rating_matrix.shape[0]):
    #     C,U,R,eigenvalues = CUR(user_rating_matrix,605,recomputeMatrix=True)  # found experimentally.. assuming equal to the rank of the matrix
    #     error = rmse(user_rating_matrix,C,U,R)
    #     if error < min_error:
    #         min_error = error
    #         min_error_index = i
    #     print(min_error,i)
    # print('min error at ',min_error,min_error_index)

    C,U,R,eigenvalues = CUR(user_rating_matrix,605,recomputeMatrix=True,energy_needed=1)
    error = rmse(user_rating_matrix,C,U,R)
    new_A = np.dot(np.dot(C,U),R)
    print('error my implementation ',error)
    
    # size_of_U = 370   # found experimentally
    # C,U,R = cur_decomposition(user_rating_matrix,size_of_U)
    # error = rmse(user_rating_matrix,C,U,R)
    # print('error library ',error)
    # new_A = np.dot(np.dot(C,U),R)
    # print(user_rating_matrix[:10,:10])
    # print(new_A[:10,:10])