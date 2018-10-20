import csv
import pickle
from collections import defaultdict

movie_csv = open("movies.csv", encoding="utf8")
rating_csv = open("ratings.csv", encoding="utf8")

movie_reader = csv.DictReader(movie_csv)
rating_reader = csv.DictReader(rating_csv)

movie_dict = defaultdict(dict)
rating_dict = defaultdict(dict)

for row in movie_reader:
    movie_dict[row['movieId']] = {row['title']:row['genres'].split('|')}

for row in rating_reader:
    rating_dict[row['userId']][row['movieId']] = row['rating']

# print(movie_dict['1'])
# print(rating_dict['1'])

movie_dict_file = open("movie_file.txt", 'wb')
rating_dict_file = open("rating_file.txt", 'wb')

pickle.dump(movie_dict, movie_dict_file)
pickle.dump(rating_dict ,rating_dict_file)

movie_csv.close()
rating_csv.close()
movie_dict_file.close()
rating_dict_file.close()
