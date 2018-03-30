import sys
import os
import pickle

from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
from pyspark.mllib.linalg import SparseVector
from pyspark.ml.evaluation import RegressionEvaluator

from FileLoader import *


class Recommender_uu():
    utility = None
    threshold = 0
    sim = None
    sim_set = None


    def __init__(self, utility, threshold=0, sim_file=None):
        self.utility = utility 
        self.threshold = threshold

        if sim_file == None or os.path.exists(sim_file) == False:
            (self.sim, self.sim_set) = self._calculateSimilarity()
            with open(sim_file, "wb") as handle:
                pickle.dump((self.sim, self.sim_set), handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(sim_file, "rb") as handle:
                (self.sim, self.sim_set) = pickle.load(handle)

        return

    def predictRating(self, user_id, movie_id, threshold=0):
        rdd = self.utility.entries
        user_cnt = self.utility.numRows()

        user_sim_set = self.sim_set[user_id]
        items = rdd.filter(lambda x: (x.i in user_sim_set) and (x.j ==movie_id) ).collect()
        #users = self.groupedRdd.filter(lambda x: x[0] in sim_set)

        rating = divider = 0
        for item in items:
            rating = rating + self.sim[(user_id, item.i)]*item.value
            divider = divider + self.sim[(user_id, item.i)]
        
        if rating==0 or divider==0:
            return 0
        else:
            return rating / divider


    def _calculateSimilarity(self, threshold=0):
        """ similar_movies: (movie_id, movie_rating_rdd)"""
        user_cnt = self.utility.numRows()
        movie_cnt = self.utility.numCols()
        rdd = self.utility.entries
        sims = dict()
        sims_set = dict()
        for i in range(1, user_cnt+1):
            sims_set[i] = list()

        users = rdd.groupBy(lambda x: x.i).collect()

        for i in range(1, user_cnt+1):
            for j in range(1, user_cnt+1):
                if i == j:
                    continue

                for user in users:
                    if user[0] == i:
                        user1 = user
                    if user[0] == j:
                        user2 = user

                vt1 = SparseVector(movie_cnt, [(user.j, 1) for user in user1[1]])
                vt2 = SparseVector(movie_cnt, [(user.j, 1) for user in user2[1]])

                sim =  vt1.dot(vt2) / (vt1.norm(2) * vt2.norm(2))
                sims[(i, j)] = sim

                sims_set[i].append(j)
        return (sims, sims_set)

def getPredictions(spark, recommender, utility, threshold=0):
    items = utility.entries.map(lambda x: (x.i, x.j)).collect()
    predictions = list()
    
    i = 0
    for item in items:
        pred = recommender.predictRating(item[0], item[1], threshold)
        predictions.append((item, pred))

        if i%10 == 0:
            print("Iteration: "+ str(i) )
        i = i+1

    return spark.sparkContext.parallize(predictions)


def main():
    movie_file = sys.argv[1]
    rating_file = sys.argv[2]

    spark = SparkSession.builder.getOrCreate()
    (training_utility, test_utility) = readRatings(spark, rating_file)

    recommender = Recommender_uu(training_utility, sim_file="./training_matrix")

    actual_rating = test_utility.entries.map(lambda x: ((x.i, x.j), x.value)).toDF(['index', 'rating'])
    predictions = getPredictions(spark, recommender, test_utility, 0).toDF(['index', 'prediction'])
    #predictions = test_utility.entries.map(lambda x: ((x.i, x.j), recommender.predictRating(x.i, x.j, 0)))

    print(actual_rating.take(10))
    #print(predictions.top(10))
    #predictions = predictions.join(actual_rating, "index")
    #evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')

    #rmse = evaluator.evaluate(predictions)
    #print(rmse)


if __name__ == "__main__":
    main()
