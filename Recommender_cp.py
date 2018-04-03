import numpy as np
import csv
import sys

from FileLoader import readRatings

from scipy import spatial
from pyspark.sql import SparkSession
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.sql import Row
   

movie_genre = { \
        "Action"    : 0,    \
        "Adventure" : 1,    \
        "Animation" : 2,    \
        'Children"s': 3,    \
        "Comedy"    : 4,    \
        "Crime"     : 5,    \
        "Documentary" : 6,  \
        "Drama"     : 7,    \
        "Fantasy"   : 8,    \
        "Film-Noir" : 9,    \
        "Horror"    : 10,   \
        "Musical"   : 11,   \
        "Mystery"   : 12,   \
        "Romance"   : 13,   \
        "Sci-Fi"    : 14,   \
        "Thriller"  : 15,   \
        "War"       : 16,   \
        "Western"   : 17,   \
        "Other"     : 18 }

movie_cnt = 164979


def readMovieChar(spark, f_name):
    my_data = list()
    with open(f_name, 'r') as handle:
        reader = csv.reader(handle, delimiter=",", quotechar='"')
        for row in reader:
            my_data.append(row)
    my_data.pop(0)

    matrix = np.zeros(shape=(int(my_data[-1][0])+1, len(movie_genre)), dtype=int)
    movie_list = dict()
    
    for movie in my_data:
        movie_id = int(movie[0])
        movie_list[movie_id] = movie[1]

        genres = movie[2].split('|')
        for each in genres:
            col_idx = movie_genre.get(each, movie_genre['Other'])
            matrix[movie_id][col_idx] = 1

    indexedRows = spark.sparkContext.parallelize([IndexedRow(i, matrix[i]) for i in range(len(matrix))])
    mat = IndexedRowMatrix(indexedRows)
    return mat, movie_list


def getUserProfile(rating_matrix, movie_matrix):
    rating_matrix_b = rating_matrix.toBlockMatrix()
    movie_matrix_b = movie_matrix.toBlockMatrix()

    return rating_matrix_b.multiply(movie_matrix_b)

def predictItem(userId, movieId, usersProfile, moviesChar, threshold=0):
    user_char = usersProfile[userId]
    movie_char = moviesChar[movieId]

    sim = 1 - spatial.distance.cosine(user_char, movie_char)
    return sim

def getRecommendation(userId, usersProfile, movies_matrix, limit=10, threshold=0):
    user_char = usersProfile.toIndexedRowMatrix().rows.filter(lambda x: x.index == userId).map(lambda x: x.vector).first()
    #print(user_char)

    movies_sim = list()
    movie_rows = movies_matrix.rows

    rank = movie_rows.filter(lambda x: np.sum(x.vector.toArray())!=0) \
            .map(lambda x: (x.index, x.vector.dot(user_char)/(x.vector.norm(2)*user_char.norm(2))) ) \
            .filter(lambda x: np.isnan(x[1])==False) \
            .sortBy(lambda x: x[1], ascending=False) \
            .take(limit)

    #print(rank)
    return rank


def getMovieNameById(rank, movie_dict):
    movie_list = list()
    for item in rank:
        movie_list.append((movie_dict[item[0]], item[1]))

    return movie_list


def main():
    movie_fname = sys.argv[1]
    rating_fname = sys.argv[2]

    spark = SparkSession.builder.getOrCreate()
    (rating_matrix, test_matrix) = readRatings(spark, rating_fname)
    (movie_matrix, movie_list) = readMovieChar(spark, movie_fname)

    #print("rating_matrix rows: " + str(rating_matrix.numRows()))
    #print("rating_matrix cols: " + str(rating_matrix.numCols()))
    #print("movie_matrix rows: " + str(movie_matrix.numRows()))
    #print("movie_matrix cols: " + str(movie_matrix.numCols()))

    user_profile = getUserProfile(rating_matrix, movie_matrix)
    #print("user_profile cols: " + str(user_profile.numRows()))
    #print("user_profile cols: " + str(user_profile.numCols()))

    ranks = getRecommendation(1, user_profile, movie_matrix)
    print(ranks)
    ranks = getMovieNameById(ranks, movie_list)
    for each in ranks:
        print(each[0] + "\t\t" + str(each[1]))


if __name__ == "__main__":
    main()
