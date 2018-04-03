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
    groupRdd = None


    def __init__(self, utility, threshold=0, sim_file=None):
        self.utility = utility 
        self.threshold = threshold
        self.groupRdd = utility.entries.groupBy(lambda x: x.i).map(lambda x: (x[0], [(each.j, each.value) for each in x[1]]))
        #print(self.groupRdd.take(3))

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
        #items = rdd.filter(lambda x: (x.i in user_sim_set) and (x.j ==movie_id) ).collect()
        #cnt = rdd.filter(lambda x: (x.i in user_sim_set) and (x.j ==movie_id) ).count()
        #rating_sum = rdd.filter(lambda x: (x.i in user_sim_set) and (x.j ==movie_id) ).map(lambda x: x.value).sum()
        candidate = self.groupRdd.filter(lambda x: x[0] in user_sim_set).map(lambda x: x[1]).filter(lambda x: movie_id in [each[0] for each in x]).flatMap(lambda x: x)
        cnt = candidate.count()
        rating_sum = candidate.map(lambda x: x[1]).sum()

        if cnt == 0:
            return 0
        else:
            return rating_sum/cnt

        #rating = divider = 0
        #for item in items:
        #    rating = rating + self.sim[(user_id, item.i)]*item.value
        #    divider = divider + self.sim[(user_id, item.i)]
        #
        #if rating==0 or divider==0:
        #    return 0
        #else:
        #    return rating / divider


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
    count = len(items)
    predictions = list()
    
    i = 0
    for item in items:
        pred = recommender.predictRating(item[0], item[1], threshold)
        predictions.append((item, pred))

        if i%10 == 0:
            print("Iteration: "+ str(i) )
            print("Complete %" + str(float(i)/count*100))
        i = i+1

    return spark.sparkContext.parallelize(predictions)

def getTopKRecommendation(predictions, movies, user_id, k):
    recomm_ids = predictions.rdd.filter(lambda x: x['index'][0] == user_id).sortBy(lambda x: x['prediction'], ascending=False)
    recomm_ids = recomm_ids.map(lambda x: int(x['index'][1])).take(k)
    recomm_movies =  movies.rdd.filter(lambda x: int(x['movieId']) in recomm_ids).map(lambda x: (int(x['movieId']), x['title']))
    movie_list = list()
    for each in recomm_ids:
        movie = recomm_movies.filter(lambda x: x[0] == each).map(lambda x: x[1]).collect() 
        movie_list.append(movie[0])
    return movie_list


def main():
    movie_file = sys.argv[1]
    rating_file = sys.argv[2]

    spark = SparkSession.builder.getOrCreate()
    (training_utility, test_utility) = readRatings(spark, rating_file, ratio=[0.80, 0.20])
    movies = readMovies(spark, movie_file)

    recommender = Recommender_uu(training_utility, sim_file="./training_matrix")

    rating = recommender.predictRating(1, 9)
    print(rating)

    actual_rating = test_utility.entries.map(lambda x: ((x.i, x.j), x.value)).toDF(['index', 'rating'])
    print(actual_rating.count())
    predictions = getPredictions(spark, recommender, test_utility, 0).toDF(['index', 'prediction'])

    #print(actual_rating.take(10))

    #print(predictions.top(10))
    predictions = predictions.join(actual_rating, "index")
    evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
    rmse = evaluator.evaluate(predictions)
    print(rmse)

    print('\n')
    movie_list = getTopKRecommendation(predictions, movies, 1, k=10)
    for each in movie_list:
        print(each)

if __name__ == "__main__":
    main()
