import numpy as np
import csv
import sys

from scipy import sparse
from numpy import genfromtxt
from sklearn.metrics.pairwise import cosine_similarity


def readRating(f_name, ratio=[0.8, 0.2], seed=0):
    # read data from csv file
    my_data = genfromtxt(f_name, delimiter=',', skip_header=True)
    my_data = np.delete(my_data, 3, 1)

    # split dataset into training and test set
    np.random.seed(seed)
    np.random.shuffle(my_data)
    cnt = int(len(my_data)*ratio[0])
    training_data, test_data = my_data[:cnt, :], my_data[cnt:, :]

    # generate matrix
    max_v = np.amax(my_data, axis=0)
    rows = int(max_v[0]+1)
    columns = int(max_v[1]+1)
    matrix = np.zeros(shape=[rows, columns], dtype=float)

    #matrix = np.full(shape=[rows, columns], fill_value=nan)
    for item in training_data:
        row = int(item[0])
        column = int(item[1])
        matrix[row][column] = item[2]

    return (matrix, test_data)



def normalize(matrix):
    #mean = np.nanmean(matrix, axis=1)
    mean = np.true_divide(matrix.sum(1),(matrix!=0).sum(1))
    normal_matrix = matrix.copy()
    for i in range(len(mean)):
        idx = (normal_matrix[i]!=0)
        normal_matrix[i][idx] = normal_matrix[i][idx] - mean[i]
        #for i in range(len(row)):
        #    if row[i] != 0:
        #        row[i] = row[i]-mean

    return normal_matrix


def getSimilarity(norm_matrix):
    sparse_matrix = sparse.csr_matrix(norm_matrix)
    similarity = cosine_similarity(sparse_matrix, dense_output=False)
    return similarity


def predictItem(userId, movieId, matrix, sim, threshold=0):
    rows, columns = matrix.shape
    
    sim_userSet = set()
    for i in range(rows):
        if(abs(sim[userId][i]-threshold) > 0.0001):
            sim_userSet.add(i)
    if(len(sim_userSet) == 0):
        return 0

    rating = 0
    divider = 0
    for sim_user in sim_userSet:
        if(sim_user!=userId and matrix[sim_user][movieId] > 0):
            rating = rating + matrix[sim_user][movieId]*sim[userId][sim_user]
            divider = divider + sim[userId][sim_user]

    if divider == 0:
        return 0
    else:
        return rating / divider

def predictTestData(test_data, matrix, sim, threshold=0):
    predictions = test_data.copy()
    for each in predictions:
        userId = int(each[0])
        movieId = int(each[1])
        each[2] = predictItem(userId, movieId, matrix, sim, threshold)

    return predictions


def main():
    f_name = sys.argv[1]
    matrix, test_data = readRating(f_name)
    print(matrix.shape)
    print(test_data.shape)

    norm_matrix = normalize(matrix)
    similarity = getSimilarity(norm_matrix)
    predictions = predictTestData(test_data, matrix, similarity.toarray(), 0)

    rmse = np.sqrt(np.mean((predictions[:][2]-test_data[:][2])**2))
    print(rmse)
    #rating = predictItem(671, 6565, matrix, similarity.toarray())
    #print(rating)
    #print(similarity.toarray())
    #print(type(similarity))
    #print(similarity.toarray()[1][199])


if __name__ == "__main__":
    main()
