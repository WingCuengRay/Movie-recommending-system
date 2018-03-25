import sys

from pyspark.sql import SparkSession
from FileLoader import readRatings
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


def main():
    movie_file = sys.argv[1]
    rating_file = sys.argv[2]

    spark = SparkSession.builder.getOrCreate()
    (training_utility, test_utility) = readRatings(spark, rating_file)

    training = training_utility.entries.map(lambda x: (x.i, x.j, x.value)).toDF(['userId', 'movieId', 'rating'])
    test = test_utility.entries.map(lambda x: (x.i, x.j, x.value)).toDF(['userId', 'movieId', 'rating'])

    als = ALS(maxIter=5, rank=70, regParam=0.01, coldStartStrategy='drop', \
            userCol='userId', itemCol='movieId', ratingCol='rating')
    als.setSeed(0)

    model = als.fit(training)
    predictions = model.transform(test)

    evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
    rmse = evaluator.evaluate(predictions)
    print("rmse of ALS recommender on test set: " + str(rmse))


if __name__ == "__main__":
    main()
