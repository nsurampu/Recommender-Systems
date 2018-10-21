import pickle

movie_pickle = open("movie_file.txt", 'rb')
rating_pickle = open("rating_file.txt", 'rb')

movie_dict = pickle.load(movie_pickle)
rating_dict  = pickle.load(rating_pickle)

movieIds = movie_dict.keys()
userIds = rating_dict.keys()

user_rating_matrix = [0] * (len(userIds) + 1)

for i in range(0, len(user_rating_matrix)):
    user_rating_matrix[i] = [0] * (193610)

for user in userIds:
    user_movies = rating_dict[user].keys()
    for movie in user_movies:
        user_rating_matrix[int(user)][int(movie)] = float(rating_dict[user][movie])

test_user = 1
pearson_similarity_dict = {}

test_user_ratings = user_rating_matrix[test_user]
total_test_ratings = sum(test_user_ratings)
mean_test_user_rating = total_test_ratings / len(rating_dict[str(test_user)])
# print(str(mean_test_user_rating))

for user in userIds:
    user = int(user)
    user_ratings = user_rating_matrix[user]
    total_user_ratings = sum(user_ratings)
    mean_user_rating = total_user_ratings / len(rating_dict[str(user)])
    temp_numerator = 0
    temp_denominator_x = 0
    temp_denominator_y = 0
    for movie in range(0, 193610):
        if (user_rating_matrix[test_user][movie] > 0) and (user_rating_matrix[user][movie] > 0):
            temp_numerator = temp_numerator + ((user_rating_matrix[test_user][movie] - mean_test_user_rating) * (user_rating_matrix[user][movie] - mean_user_rating))
            temp_denominator_x = temp_denominator_x + ((user_rating_matrix[test_user][movie] - mean_test_user_rating) ** 2)
            temp_denominator_y = temp_denominator_y + ((user_rating_matrix[user][movie] - mean_user_rating) ** 2)
    temp_denominator = (temp_denominator_x ** 0.5) * (temp_denominator_y ** 0.5)
    if temp_denominator > 0:
        coeff = temp_numerator / temp_denominator
        pearson_similarity_dict[user] = coeff

print(pearson_similarity_dict)
