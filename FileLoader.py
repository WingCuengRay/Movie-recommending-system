from pyspark.sql import SparkSession

def readMovies(f_name):
	""" Read the information about movies from f_name and return a dataframe  """
	spark = SparkSession.builder.getOrCreate()
	df = spark.read.csv(f_name, header=True)

	return df
