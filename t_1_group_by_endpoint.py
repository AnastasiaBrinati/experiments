import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, desc, sum

def group(limit, input_path, output_path):
    # Step 0: Initialize Spark session
    spark = SparkSession.builder \
        .appName("Melt and Filter RDD") \
        .getOrCreate()

    # Step 1: Load the CSV files into DataFrames
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Step 8: Count the frequency of each endpoint_uuid
    endpoint_frequency = (
        df.groupBy("endpoint_uuid")
        .agg(count("endpoint_uuid").alias("total_tasks"))
        .orderBy(desc("total_tasks"))
    )

    # Step 9: Get the top N most frequent endpoints
    top_endpoints = endpoint_frequency.limit(limit).select("endpoint_uuid").rdd.flatMap(lambda x: x).collect()
    for i in range(limit):
        # Partition the data_globus for the top N endpoints
        partition = df.filter(col("endpoint_uuid") == top_endpoints[i])
        partition = partition.drop("endpoint_uuid")
        # Save the partitions to separate CSV files
        partition.coalesce(1).write.csv(f"{output_path}/endpoint{i}", header=True, mode="overwrite")

    print(f"Partitions saved to {output_path}")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save top endpoints.")
    parser.add_argument("--limit", default=2, type=int, help="Number of top endpoints to process.")
    parser.add_argument("--input_path", default="data_globus/traces/all _endpoints_trace.csv", type=str, help="Base input path for CSV files.")
    parser.add_argument("--output_path", default="data_globus/traces", type=str, help="Base output path for CSV files.")
    args = parser.parse_args()

    group(args.limit, args.input_path, args.output_path)