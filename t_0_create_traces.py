from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import LongType
from pyspark.sql.functions import count, col, from_unixtime, first, date_format, udf, avg, desc

def create_traces():
    # Step 0: Initialize Spark session
    spark = SparkSession.builder \
        .appName("Melt and Filter RDD") \
        .getOrCreate()

    # Step 1: Load the CSV files into DataFrames
    # FUNCTIONS
    functions = spark.read.csv("data/functions.csv", header=True, inferSchema=True)
    # Drop
    functions = functions.drop('function_body_uuid')
    # TASKS
    tasks = spark.read.csv("data/tasks.csv", header=True, inferSchema=True)
    # Drop
    tasks = tasks.drop('anonymized_user_uuid')

    # Step 2: Perform an inner join to merge the DataFrames based on the specified columns
    df_joined = tasks.join(functions, on=['function_uuid'], how='inner')

    # Step 3: Drop NaN values
    df_clean = df_joined.dropna(how='any')

    # Step 5: Convert the DataFrame to an RDD and proceed with columns refactoring
    rdd = df_clean.rdd

    # Use flatMap to melt the DataFrame, transforming it from wide to long format
    melted_rdd = rdd.flatMap(lambda row: [
        Row(
            # from tasks
            arrival_timestamp = row['received'],

            endpoint_uuid=row['endpoint_uuid'],
            argument_size=row['argument_size'],

            # from functions
            loc=row['loc'],
            cyc_complexity=row['cyc_complexity'],
            num_of_imports=row['num_of_imports'],
        )
    ])

    # Step 6: Convert the transformed RDD back to a DataFrame
    filtered_df = spark.createDataFrame(melted_rdd)
    df = filtered_df.dropna(how='any')

    # more Ordering
    result = df.orderBy("arrival_timestamp")

    # save
    result.coalesce(1).write.csv("data/traces", header=True, mode="overwrite")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    create_traces()