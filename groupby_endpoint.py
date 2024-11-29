from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, desc

def group():
    # Step 0: Initialize Spark session
    spark = SparkSession.builder \
        .appName("Melt and Filter RDD") \
        .getOrCreate()

    # Step 1: Load the CSV files into DataFrames
    df = spark.read.csv("data/globus/globus.csv", header=True, inferSchema=True)

    # Step 8: Count the frequency of each endpoint_uuid
    endpoint_frequency = (
        df.groupBy("endpoint_uuid")
        .agg(count("*").alias("frequency"))
        .orderBy(desc("frequency"))
    )

    # How many endpoints? N
    limit = 10

    # Step 9: Get the top N most frequent endpoints
    top_endpoints = endpoint_frequency.limit(limit).select("endpoint_uuid").rdd.flatMap(lambda x: x).collect()
    for i in range(limit):
       # Partition the data for the top N endpoints
       partition = df.filter(col("endpoint_uuid") == top_endpoints[i])
       # Save the partitions to separate CSV files
       partition.coalesce(1).write.csv("data/endpoints/endpoint" + str(i), header=True, mode="overwrite")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    group()