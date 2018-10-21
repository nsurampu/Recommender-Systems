import pickle
import math

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

test_user = 1
test_user_rating = 0
pearson_dict = {}
last_movie = int(list(rating_dict[str(test_user)].keys())[-1]) + 1

# print(str(mean_test_user_rating))

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

# print(pearson_dict)
sorted_pearson_dict = {t: pearson_dict[t] for t in sorted(pearson_dict, key=pearson_dict.get, reverse=True)}

top_matches = {k:sorted_pearson_dict[k] for k in list(sorted_pearson_dict)[:6]}

print(top_matches)

test_user_ratings = user_rating_matrix[test_user]
movie_pred_dict = {}
