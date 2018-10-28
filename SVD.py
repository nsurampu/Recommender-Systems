import pickle
import math
import numpy as np
np.set_printoptions(threshold=np.inf)

def Energy(A):
    A = A*A
    return A.sum()

def RMSE(user_array,FinalA):
    error = user_array - FinalA
    # error = error[1:,:]
    sqerror = error*error
    # print(sqerror.size)
    RMSE = sqerror.sum()/(sqerror.size)
    RMSE = math.sqrt(RMSE)
    return RMSE

def SVD(user_array):
    UAT = user_array.T
    array_AAT = np.dot(user_array, (UAT))
    eigenvalues, eigenvectors_AAT = np.linalg.eig(array_AAT)
    eigenvectors_AAT = eigenvectors_AAT.real
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors_AAT = eigenvectors_AAT[:,idx]

    eigenvalues[eigenvalues < 1.0e-10] = 0
    #Finding rank
    rank = 0
    for i in eigenvalues:
        rank = rank + 1
        if(i.imag != 0):
            break
    rank = rank -1
    rank = 610
    print("Rank: ",rank)
    # Rank = Row that contains last non zero value
    #Reducing size of eigenvalues to only include the actual rank
    eigenvalues = eigenvalues[0:(rank-1)]
    # eigenvalues = eigenvalues.real
    #Build sigma
    sigma = np.diag(eigenvalues)
    #U of SVD with
    U = eigenvectors_AAT
    array_ATA = np.dot((UAT), user_array)
    eigenvalues_irr, eigenvectors_ATA = np.linalg.eig(array_ATA)
    idx = eigenvalues_irr.argsort()[::-1]
    eigenvalues_irr = eigenvalues_irr[idx]
    eigenvectors_ATA = eigenvectors_ATA[:,idx]
    V = eigenvectors_ATA
    # Slicing to match Size
    U = U[:, 0:rank-1]
    V = V[:, 0:rank-1]
    # print(sigma)
    sigma = np.sqrt(sigma)
    print("Size of U,V",U.shape, V.shape)
    return U,sigma,V,eigenvalues

def Query(q,V):
    #Calculates the query result given a
    temp = np.dot(q,V)
    final = np.dot(temp,V.T)
    return final

def Precision_top_k(k,q,V):
    final = Query(q,V)
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
# U = U.real
# V = V.real
new_A = np.dot(U,sigma)
FinalA = np.dot(new_A,VT)

print(FinalA.shape)

u,s,v = np.linalg.svd(user_array, full_matrices=True)
# print(u)
sn = np.diag(s)
n = np.dot(u,sn)
print("SVD Funct",u.shape,n.shape,sn.shape,v.shape)
v = v[:,0:610]
al = np.dot(n,v.T)
print("SVD FUNCT",RMSE(user_array,al))


energy = Energy(eigenvalues)
# Reverse the eigenvalue np array
Reduction_array = np.empty([1])
for i in range(eigenvalues.size,0,-1):
    temp = eigenvalues[0:i]
    temp_Energy = Energy(temp)
    if(temp_Energy >= 0.9 * energy):
        Reduction_array = temp
    else:
        break    


size = Reduction_array.size
print(size)
Reduction_array = Reduction_array[0:(size-1)]
Reduction_array = Reduction_array.real
sigma_reduced = np.diag(Reduction_array)
U_reduced = U[:,0:(size-1)]
V_reduced = V[:,0:(size-1)]
VT_reduced = V_reduced.T
new_A_reduced = np.dot(U_reduced,sigma_reduced)
ReducedA = np.dot(new_A_reduced,VT_reduced)


print("Non reduced",RMSE(user_array,FinalA))
print("90% reduced",RMSE(user_array,ReducedA))

U_file = open("U_file.txt", 'wb')
V_file = open("V_file.txt", 'wb')
sigma_file = open("sigma_file.txt", 'wb')
U_reduced_file = open("U_reduced_file.txt", 'wb')
V_reduced_file = open("V_reduced_file.txt", 'wb')
sigma_reduced_file = open("sigma_reduced_file.txt", 'wb')

user_map = np.dot(U,sigma)
sigma_map = np.dot(sigma,V.T)



pickle.dump(U, U_file)
pickle.dump(V, V_file)
pickle.dump(sigma ,sigma_file)

U_file.close()
V_file.close()
sigma_file.close()


pickle.dump(U_reduced, U_reduced_file)
pickle.dump(V_reduced, V_reduced_file)
pickle.dump(sigma_reduced ,sigma_reduced_file)

U_reduced_file.close()
V_reduced_file.close()
sigma_reduced_file.close()
