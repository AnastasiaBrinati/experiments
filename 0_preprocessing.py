from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import LongType
from pyspark.sql.functions import count, col, from_unixtime, first, date_format, udf, avg, desc

def process():
    # Step 0: Initialize Spark session
    spark = SparkSession.builder \
        .appName("Melt and Filter RDD") \
        .getOrCreate()

    # Step 1: Load the CSV files into DataFrames
    # ENDPOINTS
    endpoints = spark.read.csv("data/globus/endpoints.csv", header=True, inferSchema=True)
    # FUNCTIONS
    functions = spark.read.csv("data/globus/functions.csv", header=True, inferSchema=True)
    # Drop
    functions = functions.drop('function_body_uuid')
    # TASKS
    tasks = spark.read.csv("data/globus/tasks.csv", header=True, inferSchema=True)
    # Drop
    tasks = tasks.drop('anonymized_user_uuid')

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
            arrival_timestamp = f"{row['arrival_timestamp']}",

            #function_uuid=row['function_uuid'],
            endpoint_uuid=row['endpoint_uuid'],
            #task_uuid=row['task_uuid'],

            assignment_time=row['waiting_for_nodes'] - row['received'], # time between the task arriving to the cloud platform the task being received by an endpoint
            scheduling_time=row['waiting_for_launch'] - row['waiting_for_nodes'], # time between the task received by an endpoint and the task assigned to a worker
            queue_time=row['execution_start'] - row['waiting_for_launch'], # time between the task assigned to a worker and the execution start
            execution_time=row['execution_end'] - row['execution_start'], # time between the task execution start and end
            results_time=row['result_received'] - row['execution_end'],
            total_execution_time=row['result_received'] - row['received'], # time between the task arriving and the results.csv being reported, to the could platform

            argument_size=row['argument_size'],

            # from functions
            loc=row['loc'],
            cyc_complexity=row['cyc_complexity'],
            num_of_imports=row['num_of_imports'],

            # from endpoints
            **{f"e_type_{e_type.replace('.', '_')}": 1 if row['endpoint_type'] == e_type else 0 for e_type in unique_endpoint_type},  # One-hot encoding for all unique endpoint types
            **{f"e_vers_{e_vers.replace('.', '_')}": 1 if row['endpoint_version'] == e_vers else 0 for e_vers in unique_endpoint_version}  # One-hot encoding for all unique endpoint versions
        )
    ])

    # Step 6: Convert the transformed RDD back to a DataFrame
    filtered_df = spark.createDataFrame(melted_rdd)
    df = filtered_df.dropna(how='any')

    # Select columns relative to endpoint types
    endpoint_type_columns = [c for c in df.columns if c.startswith("e_type_")]

    # Step 7: Grouping and calculations
    result = (
        df.groupBy("endpoint_uuid", "arrival_timestamp")
        .agg(
            count("*").alias("n_invocations"),  # Numero di invocazioni
            avg("loc").alias("avg_loc"),  # Media delle linee di codice
            avg("cyc_complexity").alias("avg_cyc_complexity"),  # Media della complessità ciclica
            avg("num_of_imports").alias("avg_num_of_imports"),  # Media del numero di importazioni
            avg("argument_size").alias("avg_argument_size"),  # Media della dimensione degli argomenti
            *[first(col(c)).alias(c) for c in endpoint_type_columns],  # Mantenimento delle colonne `e_type_*`
            #avg("total_execution_time").alias("avg_total_execution_time"),  # Media del tempo di esecuzione
            #avg("scheduling_time").alias("avg_scheduling_time")  # Media del tempo di scheduling
        )
    )

    # more Ordering
    result = result.orderBy("arrival_timestamp")

    # save
    result.coalesce(1).write.csv("data/globus", header=True, mode="overwrite")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    process()
    # skipping scaling for now
    # skipping interpolation as well because task is on bigger scale (just fill with 0 where no elements)