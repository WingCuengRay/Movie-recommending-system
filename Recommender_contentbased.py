import numpy as np
import csv
import sys

from scipy import spatial

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

def readRating(f_name, ratio=[0.8, 0.2], seed=0):
    # read data from csv file
    my_data = np.genfromtxt(f_name, delimiter=',', skip_header=True)
    my_data = np.delete(my_data, 3, 1)

    # split dataset into training and test set
    np.random.seed(seed)
    np.random.shuffle(my_data)
    cnt = int(len(my_data)*ratio[0])
    training_data, test_data = my_data[:cnt, :], my_data[cnt:, :]

    # generate matrix
    max_v = np.amax(my_data, axis=0)
    rows = int(max_v[0]+1)
    columns = int(movie_cnt+1)
    matrix = np.zeros(shape=[rows, columns], dtype=float)

    #matrix = np.full(shape=[rows, columns], fill_value=nan)
    for item in training_data:
        row = int(item[0])
        column = int(item[1])
        matrix[row][column] = item[2]

    return (matrix, test_data)


def readMovieChar(f_name):
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

    return matrix, movie_list


def getUserProfile(utility, movie_char):
    usersProfile = utility.dot(movie_char)
    return usersProfile

def predictItem(userId, movieId, usersProfile, moviesChar, threshold=0):
    user_char = usersProfile[userId]
    movie_char = moviesChar[movieId]

    sim = 1 - spatial.distance.cosine(user_char, movie_char)
    return sim

def getRecommendation(userId, usersProfile, moviesChar, limit=10, threshold=0):
    user_char = usersProfile[userId]
    movies_sim = list()
    for i in range(len(moviesChar)):
        sim = 1 - spatial.distance.cosine(user_char, moviesChar[i])
        if sim > threshold:
            movies_sim.append((i, sim))

    limit = limit if limit<len(movies_sim) else len(movies_sim)
    movies_sim = sorted(movies_sim, key=lambda x: x[1], reverse=True)

    return movies_sim[:limit]


def getMovieNameById(recommendations, movie_dict):
    movie_list = list()
    for item in recommendations:
        movie_list.append(movie_dict[item[0]])

    return movie_list


def main():
    movie_fname = sys.argv[1]
    rating_fname = sys.argv[2]

    (utility, test_data) = readRating(rating_fname)
    (movie_char, movie_dict) = readMovieChar(movie_fname)
    usersProfile = getUserProfile(utility, movie_char)

    print(utility.shape)
    print(movie_char.shape)
    print(usersProfile.shape)
    print(usersProfile)

    print('---------prediction-------------')
    result = predictItem(1, 31, usersProfile, movie_char)
    print(result)
    recommendations = getRecommendation(1, usersProfile, movie_char)
    movie_list = getMovieNameById(recommendations, movie_dict)
    print(movie_list)


if __name__ == "__main__":
    main()
