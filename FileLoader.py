import sys 
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.mllib.linalg import Matrix, Matrices
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
from pyspark.sql import Row


def readMovies(spark, f_name):
    """ Read the information about movies from f_name and return a dataframe  """
    rdd = spark.read.csv(f_name, header=True).rdd
    rdd = rdd.map(lambda row: (row['movieId'], row['title'], row['genres'].split('|')))

    return rdd.toDF(['movieId', 'title', 'genres'])

def normalize(spark, df):
    rdd = df.rdd
    total = rdd.map(lambda row: (row['userId'], float(row['rating']))).reduceByKey(lambda x,y : x+y)
    cnt = rdd.map(lambda row: (row['userId'], 1)).reduceByKey(lambda x,y: x+y)
    #cnt = rdd.countByKey()

    user_mean = total.join(cnt).map(lambda x: (x[0], x[1][0]/x[1][1])).collectAsMap()
    data = rdd.collect()

    for i in range(len(data)):
        rating = float(data[i][2])
        if rating != 0:
            norm_rating = rating - user_mean[data[i][0]]
            data[i] = Row(userId=data[i][0], movieId=data[i][1], rating=norm_rating)

    new_df = spark.sparkContext.parallelize(data).toDF()

    return new_df


    """
    rating_mean = df.groupBy(df.userId).agg(avg('rating').alias('rating-mean'))

    df = df.join(rating_mean, 'userId')
    df.
    df.withColumn('rating-norm', df['rating']-df['rating-mean'])
    """


def readRatings(spark, f_name, ratio=[0.8, 0.2], seed=0):
    """ Read the rating of users for movies 
        Return the utility matrix"""
    df = spark.read.csv(f_name, header=True)
    #df = normalize(spark, df)
    rdd = df.rdd

    (training, test) = df.randomSplit(ratio, seed=seed)
    
    training_utility = CoordinateMatrix(training.rdd.map(lambda row: MatrixEntry(row['userId'], row['movieId'], row['rating'])))
    test_utility = CoordinateMatrix(test.rdd.map(lambda row: MatrixEntry(row['userId'], row['movieId'], row['rating'])))
    
    return (training_utility, test_utility)



def main():
    movie_file = sys.argv[1]
    rating_file = sys.argv[2]

    #movie_df = readMovies(movie_file)
    #rating_df = readRatings(rating_file)

    #movie_df.show()
    #rating_df.show()

    conf = SparkConf().setAppName("App")
    conf = (conf.setMaster('local[*]') \
        .set('spark.executor.memory', '4G') \
        .set('spark.driver.memory', '45G') \
        .set('spark.driver.maxResultSize', '10G'))
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    spark = sqlContext.sparkSession
    
    (utility, test_utility) = readRatings(spark, rating_file)
    rdd = utility.entries

    print(utility.numRows())
    print(utility.numCols())
    print(rdd.top(10))

    print('\n\n')
    cs = test_utility.toBlockMatrix().toLocalMatrix()
    print(cs)

if __name__ == "__main__":
    main()
