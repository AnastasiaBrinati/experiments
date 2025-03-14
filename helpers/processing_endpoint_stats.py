from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import LongType
from pyspark.sql.functions import count, col, from_unixtime, first, date_format, udf, avg, expr, desc, unix_timestamp, min, max, countDistinct

def process():
    # Step 0: Initialize Spark session
    spark = SparkSession.builder \
        .appName("Melt and Filter RDD") \
        .getOrCreate()

    # Step 1: Load the CSV files into DataFrames
    # ENDPOINTS
    endpoints = spark.read.csv("data_globus/endpoints.csv", header=True, inferSchema=True)
    # FUNCTIONS
    functions = spark.read.csv("data_globus/functions.csv", header=True, inferSchema=True)
    # Drop
    #functions = functions.drop('function_body_uuid')
    # TASKS
    tasks = spark.read.csv("data_globus/tasks.csv", header=True, inferSchema=True)
    # Drop
    #tasks = tasks.drop('anonymized_user_uuid')

    # Step 2: Perform an inner join to merge the DataFrames based on the specified columns
    tasks_joined = tasks.join(endpoints, on=['endpoint_uuid'], how='inner')
    df_joined = tasks_joined.join(functions, on=['function_uuid'], how='inner')

    # Step 3: Drop NaN values
    df_clean = df_joined.dropna(how='any')

    # Step 4: Convert from nanoseconds to seconds
    df = df_clean.withColumn("arrival_timestamp", date_format(from_unixtime((col("received") / 1_000_000_000).cast(LongType())), "yyyy-MM-dd HH:mm:ss"))

    # Step 5: Convert the DataFrame to an RDD and proceed with columns refactoring
    rdd = df.rdd
    unique_endpoint_type = rdd.map(lambda row: row['endpoint_type']).distinct().collect()
    unique_endpoint_version = rdd.map(lambda row: row['endpoint_version']).distinct().collect()
    # Use flatMap to melt the DataFrame, transforming it from wide to long format
    melted_rdd = rdd.flatMap(lambda row: [
        Row(
            # from tasks
            timestamp = f"{row['arrival_timestamp']}",

            endpoint_uuid=row['endpoint_uuid'],
            user_uuid=row['anonymized_user_uuid'],
            function_uuid=row['function_uuid'],
            task_uuid=row['task_uuid'],

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

    # Step 7: Grouping and calculations
    result = (
        df.groupBy("endpoint_uuid")
        .agg(
            count("task_uuid").alias("n_tasks"),  # Numero di invocazioni
            countDistinct("user_uuid").alias("n_users"),
            countDistinct("function_uuid").alias("n_functions"),
            #avg("loc").alias("avg_loc"),  # Media delle linee di codice
            #avg("cyc_complexity").alias("avg_cyc_complexity"),  # Media della complessità ciclica
            #avg("num_of_imports").alias("avg_num_of_imports"),  # Media del numero di importazioni
            #avg("argument_size").alias("avg_argument_size"),  # Media della dimensione degli argomenti
            min("timestamp").alias("first_function_call"),  # Earliest timestamp
            max("timestamp").alias("last_function_call"),  # Latest timestamp
            (unix_timestamp(max("timestamp")) - unix_timestamp(min("timestamp"))).alias("time_range_seconds"), # Range temporale in secondi
        )
        .withColumn(
            "function_rate_per_minute",
            expr("n_tasks / (time_range_seconds / 60)")  # Calcolo del rate per minuto
        )
    )

    # Step 8: Ordering by number of invocations
    result = result.drop("time_range_seconds")
    result = result.orderBy(desc('n_tasks'))

    result.show()

    # Ora voglio fare un altro df o rdd in cui raccolgo sempre per endpoint però vado ad usare una finestra per raccogliere

    # save
    result.coalesce(1).write.csv("data_globus/globus/prove", header=True, mode="overwrite")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    process()
    # skipping scaling for now
    # skipping interpolation as well because task is on bigger scale (just fill with 0 where no elements)