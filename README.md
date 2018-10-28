# Recommender Systems

This project implements various types of recommender system techniques using Python3.

### Description

This project implements 3 primary types of recommender system techniques: Collaborative filtering, SVD and CUR matrix decompositions. Each one additionally also implements
the global baseline technique. The systems are evaluated using RMSE, precision on top K, and Spearman ranking.

### Prerequisites

Python3 along with nltk library is required to run this program.

### Structure of project

data_processing.py : Access the movie and ratings datasets and process them for usage in the various recommender systems.

collaborative_filtering.py : Implements the collaborative filtering recommender model. It is divided into 3 parts- precision on top K, Spearman ranking and RMSE
1. Precision on top K uses Pearson correlation for checking similarity between the test user and the other users and accordingly predicts ratings. It then calculates the precision among the top ratings.
2. Spearman ranking uses the Spearman ranking technique to predict the rating of a test user for some test movie.
3. RMSE uses Pearson correlation to calculate the similarities and then calculates the RMSE over some test dataset, which consists of randomly selected users accounting for
20% of the entire user database. It then calculates the RMSE from all the predicted ratings and displays a final RMSE value.

### Run this in your project folder:

$ python data_processing.py </br>
$ python collaborative_filtering.py

### Results during runs

System | RMSE | Precision on top K | Spearman Ranking
--- | --- | --- | --- | ---
Collaborative filtering | 0.9 | 72% | 40%
Collaborative filtering (baseline) | 3.6 | 65% | 26%
SVD | | |
SVD (90% Energy) | | |
CUR | | |
CUR (90% Energy) | | | 

### Built With

The project uses

    Python3
    Numpy
    Scipy

### Authors

Chandrahas Aroori [https://github.com/Exorust] </br>
Naren Surampudi [https://github.com/nsurampu] </br>
Aditya Srikanth [https://github.com/aditya-srikanth]

### Acknowledgments

We'd like to thank our Information Retrieval instructor to give us this opportunity to make such a project.
