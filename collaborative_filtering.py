import pickle
import math
import scipy.stats as ss

movie_pickle = open("movie_file.txt", 'rb')
rating_pickle = open("rating_file.txt", 'rb')

movie_dict = pickle.load(movie_pickle)
rating_dict  = pickle.load(rating_pickle)

movieIds = movie_dict.keys()
userIds = rating_dict.keys()
last_movieId = int(list(movieIds)[-1])

user_rating_matrix = [0] * (len(userIds) + 1)

for i in range(0, len(user_rating_matrix)):
    user_rating_matrix[i] = [0] * (last_movieId + 1)

for user in userIds:
    user_movies = rating_dict[user].keys()
    for movie in user_movies:
        user_rating_matrix[int(user)][int(movie)] = float(rating_dict[user][movie])

test_user = int(input("Enter userId: "))
test_movie = int(input("Enter movieId: "))
method = int(input("Enter method: 1. K-neigbours 2. Spearman Ranking: "))
test_user_movies_dict = rating_dict[str(test_user)]
test_user_movies = test_user_movies_dict.keys()
mean_test_rating = 0
test_user_rating = 0
pearson_dict = {}
last_movie = int(list(rating_dict[str(test_user)].keys())[-1]) + 1

# Baseline estimation

# K-neighbours method
if method == 1:
    for user in userIds:
        user = int(user)
        user_rating = 0
        temp_numerator = 0
        temp_denominator_x = 0
        temp_denominator_y = 0
        length = 0
        for movie in range(0, last_movie):
            if (user_rating_matrix[test_user][movie] > 0) and (user_rating_matrix[user][movie] > 0):
                test_user_rating = test_user_rating + user_rating_matrix[test_user][movie]
                user_rating = user_rating = user_rating + user_rating_matrix[user][movie]
                length = length + 1
        if length > 0:
            mean_test_user_rating = test_user_rating / length
            mean_user_rating = user_rating / length
            for movie in range(1, last_movie):
                if (user_rating_matrix[test_user][movie] > 0) and (user_rating_matrix[user][movie] > 0):
                    temp_numerator = temp_numerator + ((user_rating_matrix[test_user][movie] - mean_test_user_rating) * (user_rating_matrix[user][movie] - mean_user_rating))
                    temp_denominator_x = temp_denominator_x + ((user_rating_matrix[test_user][movie] - mean_test_user_rating) ** 2)
                    temp_denominator_y = temp_denominator_y + ((user_rating_matrix[user][movie] - mean_user_rating) ** 2)
            temp_denominator = math.sqrt(temp_denominator_x) * math.sqrt(temp_denominator_y)
            if temp_denominator > 0:
                coeff = temp_numerator / temp_denominator
                pearson_dict[user] = coeff

    sorted_pearson_dict = {t: pearson_dict[t] for t in sorted(pearson_dict, key=pearson_dict.get, reverse=True)}

    top_matches = {k:sorted_pearson_dict[k] for k in list(sorted_pearson_dict)[:5]}

    top_users = list(top_matches.keys())

    temp_numerator = 0
    temp_denominator = 0

    for user in top_users:
        if top_matches[user] != 1:
            temp_numerator = temp_numerator + (float(top_matches[user]) * user_rating_matrix[user][test_movie])
            temp_denominator = temp_denominator + float(top_matches[user])

    pred_rating = round((temp_numerator / temp_denominator), 2)
    test_rating = user_rating_matrix[test_user][test_movie]
    print("Predicted rating: " + str(pred_rating))
    if test_rating > 0:
        k_error = abs(pred_rating - test_rating) * (100 / (test_rating))
        print("Actual rating: " + str(test_rating))
        print("Closeness for k-neigbours: " + str(100 - round(k_error, 2)) + "%")

# Spearman Method
elif method == 2:
    temp_test_rating_dict = {}
    temp_user_rating_dict = {}
    for user in userIds:
        user = int(user)
        some_list_1 = []    # for test user
        some_list_2 = []    # for other user
        for movie in range(0, last_movie):
            if (user_rating_matrix[test_user][movie] > 0) and (user_rating_matrix[user][movie] > 0):
                some_list_1.append((str(user_rating_matrix[test_user][movie])))
                some_list_2.append(str(user_rating_matrix[user][movie]))
        temp_test_rating_dict[str(user)] = some_list_1
        temp_user_rating_dict[str(user)] = some_list_2

    test_movies_ranks = {}
    user_movies_ranks = {}

    for user in temp_user_rating_dict.keys():
        test_movies_ranks[user] = ss.rankdata([-1 * user for user in temp_test_rating_dict[user]])
        user_movies_ranks[user] = ss.rankdata([-1 * user for user in temp_user_rating_dict[user]])

    spearman_dict = {}
    users = test_movies_ranks.keys()
    sq_d = 0
    for rank in range(0, len(temp_test_rating_dict['3'])):
        sq_d = sq_d + (test_movies_ranks['3'][rank] - user_movies_ranks['3'][rank]) ** 2
        result = 1 - ((6 * sq_d) / (len(temp_test_rating_dict['3']) * ((len(temp_test_rating_dict['3']) ** 2))))

    for user in users:
        sq_d = 0
        if len(temp_test_rating_dict[user]) > 0:
            for rank in range(0, len(temp_test_rating_dict[user])):
                sq_d = sq_d + (test_movies_ranks[user][rank] - user_movies_ranks[user][rank]) ** 2
            result = 1 - ((6 * sq_d) / (len(temp_test_rating_dict[user]) * ((len(temp_test_rating_dict[user]) ** 2))))
            spearman_dict[user] = str(result)

    temp_numerator = 0
    temp_denominator = 0

    for user in users:
        if len(temp_test_rating_dict[user]) > 0:
            if user is not str(test_user) and abs(float(spearman_dict[user])) > 0.35:
                temp_numerator = temp_numerator + (float(spearman_dict[user]) * user_rating_matrix[int(user)][test_movie])
                temp_denominator = temp_denominator + float(spearman_dict[user])

    pred_rating = round((temp_numerator / temp_denominator), 2)
    test_rating = user_rating_matrix[test_user][test_movie]
    print("Predicted rating: " + str(pred_rating))
    if test_rating > 0:
        spearman_error = abs(pred_rating - test_rating) * (100 / (test_rating))
        print("Actual rating: " + str(test_rating))
        print("Closeness for Spearman ranking: " + str(100 - round(spearman_error, 2)) + "%")
