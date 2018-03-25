import sys
import pickle

from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
from  pyspark.mllib.linalg import SparseVector

from FileLoader import *


class Recommender_uu():
    utility = None
    threshold = 0
    sim = None


    def __init__(self, utility, threshold=0, sim_file=None):
        self.utility = utility 
        self.threshold = threshold

        if sim_file == None:
            self.sim = self._calculateSimilarity()
            with open("sim_matrix", "wb") as handle:
                pickle.dump(self.sim, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(sim_file, "rb") as handle:
                self.sim = pickle.load(handle)

        return

    def predictRating(self, user_id, movie_id, threshold=0):
        rdd = self.utility.entries
        user_cnt = self.utility.numRows()

        sim_set = set()
        for i in range(1, user_cnt+1):
            if i!=user_id and self.sim[(user_id, i)]!=None and self.sim[(user_id, i)]>=threshold:
                sim_set.add(i)

        items = rdd.filter(lambda x: (x.i in sim_set) and (x.j ==movie_id) ).collect()
        rating = divider = 0
        for item in items:
            rating = rating + self.sim[(user_id, item.i)]*item.value
            divider = divider + self.sim[(user_id, item.i)]
        
        return rating / divider

    def f(x):
        print(x)


    def _calculateSimilarity(self):
        """ similar_movies: (movie_id, movie_rating_rdd)"""
        user_cnt = self.utility.numRows()
        movie_cnt = self.utility.numCols()
        rdd = self.utility.entries
        sims = dict()

        users = rdd.groupBy(lambda x: x.i).collect()

        for i in range(1, user_cnt+1):
            for j in range(1, user_cnt+1):
                for user in users:
                    if user[0] == i:
                        user1 = user
                    if user[0] == j:
                        user2 = user

                vt1 = SparseVector(movie_cnt, [(user.j, 1) for user in user1[1]])
                vt2 = SparseVector(movie_cnt, [(user.j, 1) for user in user2[1]])

                sim =  vt1.dot(vt2) / (vt1.norm(2) * vt2.norm(2))
                sims[(i, j)] = sim
        return sims



def main():
    movie_file = sys.argv[1]
    rating_file = sys.argv[2]

    spark = SparkSession.builder.getOrCreate()
    utility = readRatings(spark, rating_file)

    recommender = Recommender_uu(utility, sim_file="./sim_matrix")
    rating = recommender.predictRating(1, 1405, threshold=0.2)
    print(rating)




if __name__ == "__main__":
    main()
