import pickle
import math
import numpy as np
import os
np.set_printoptions(threshold=np.inf)

def CUR(user_rating_matrix,recomputeMatrix=False):
    
    file = 'CUR_Matrices.txt'

    if recomputeMatrix or not os.path.exists(file):
        # prob dist along column
        p_column = np.power(np.sum(user_rating_matrix,axis=0),2)/np.power(np.sum(user_rating_matrix),2)
        num_columns = 607 # found experimentally.. assuming equal to the rank of the matrix
        column_counter = 0 # number of columns added to the array
        column_repetition = dict()
        column_positions = dict()

        for i in range(user_rating_matrix.shape[1]):
            column_repetition[i] = 0 

        while column_counter < num_columns:
            candidate_column = np.argmax(p_column)
            if column_repetition[candidate_column] == 0:
                temp = np.reshape(user_rating_matrix[:,candidate_column],(user_rating_matrix.shape[0],1))
                temp = temp/np.sqrt(num_columns*p_column[candidate_column])
                if column_counter == 0:
                    C = temp
                elif column_counter == 1: # first column or second column have issues
                    C = np.reshape(C,(user_rating_matrix.shape[0],1))
                    C = np.hstack((C,temp))
                else:
                    C = np.hstack((C,temp))
                column_repetition[candidate_column] += 1
                column_positions[candidate_column] = column_counter
                column_counter += 1
                continue

            elif column_repetition[candidate_column] >= 1:
                position_of_column = column_positions[candidate_column]
                column_repetition[candidate_column] += 1
                p_column[candidate_column] = p_column[candidate_column]/np.sqrt(column_repetition[candidate_column])
                multiply = np.ones(C.shape)
                multiply[:,position_of_column] = np.sqrt(column_repetition[candidate_column])
                C = C*multiply
        print('C computed')        

        num_rows = 607 # found experimentally
        p_rows = np.power(np.sum(user_rating_matrix,axis=1),2)/np.power(np.sum(user_rating_matrix),2)
        row_counter = 0
        row_repetition = dict()
        row_positions = dict()
    

        for i in range(user_rating_matrix.shape[0]):
            row_repetition[i] = 0

        while row_counter < num_rows:
            candidate_row = np.argmax(p_rows)
            if row_repetition[candidate_row] == 0:
                temp = np.reshape(user_rating_matrix[candidate_row,:],(1,user_rating_matrix.shape[1]))
                temp = temp/np.sqrt(num_rows*p_rows[candidate_row])
                if row_counter == 0:
                    R = temp
                elif row_counter == 1: # first column or second column have issues
                    R = np.reshape(R,(1,user_rating_matrix.shape[1]))
                    R = np.vstack((R,temp))
                else:
                    R = np.vstack((R,temp))
                row_repetition[candidate_row] += 1
                row_positions[candidate_row] = row_counter
                row_counter += 1
                continue

            elif row_repetition[candidate_row] >= 1:
                position_of_row = row_positions[candidate_row]
                row_repetition[candidate_row] += 1
                p_rows[candidate_row] = p_rows[candidate_row]/np.sqrt(row_repetition[candidate_row])
                multiply = np.ones(R.shape)
                multiply[position_of_row,:] = np.sqrt(row_repetition[candidate_row])
                R = R*multiply
        print('R computed')

        # compute U
        


        # save file
        with open(file,'wb') as f:
            data = {}
            data['C'] = C
            data['R'] = R
            pickle.dump(data,f)
    else:
        with open(file,'rb') as f:
            data = pickle.load(f) 
            print('done')
            C = data['C']
            R = data['R']
            print(C.shape,R.shape)       

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
    CUR(user_rating_matrix,recomputeMatrix=True)