from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, from_unixtime, first, date_format, udf, avg, desc, window
from collections import defaultdict
from datasets import Dataset, DatasetDict

def time_window(input_csv, output_csv):
    # input_csv: file csv to group

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Melt and Filter RDD") \
        .getOrCreate()

    df = spark.read.csv(input_csv, header=True, inferSchema=True)

    # Use window function to group by T-second intervals
    time_window_30 = window(col("arrival_timestamp").cast("timestamp"), f"{30} seconds")
    time_window_60 = window(col("arrival_timestamp").cast("timestamp"), f"{60} seconds")
    time_window_300 = window(col("arrival_timestamp").cast("timestamp"), f"{300} seconds")

    # Grouping and calculations
    endpoint_type_columns = [c for c in df.columns if c.startswith("e_type_")]
    result_30 = (
        df.groupBy(time_window_30)
        .agg(
            count("*").alias("n_invocations"),  # Numero di invocazioni
            avg("avg_loc").alias("avg_loc"),  # Media delle linee di codice
            avg("avg_cyc_complexity").alias("avg_cyc_complexity"),  # Media della complessità ciclica
            avg("avg_num_of_imports").alias("avg_num_of_imports"),  # Media del numero di importazioni
            avg("avg_argument_size").alias("avg_argument_size"),  # Media della dimensione degli argomenti
            *[first(col(c)).alias(c) for c in endpoint_type_columns],  # Mantenimento delle colonne `e_type_*`
            # avg("total_execution_time").alias("avg_total_execution_time"),  # Media del tempo di esecuzione
            # avg("scheduling_time").alias("avg_scheduling_time")  # Media del tempo di scheduling
        )
    )

    # Grouping and calculations
    result_60 = (
        df.groupBy(time_window_60)
        .agg(
            count("*").alias("n_invocations"),  # Numero di invocazioni
            avg("avg_loc").alias("avg_loc"),  # Media delle linee di codice
            avg("avg_cyc_complexity").alias("avg_cyc_complexity"),  # Media della complessità ciclica
            avg("avg_num_of_imports").alias("avg_num_of_imports"),  # Media del numero di importazioni
            avg("avg_argument_size").alias("avg_argument_size"),  # Media della dimensione degli argomenti
            *[first(col(c)).alias(c) for c in endpoint_type_columns],  # Mantenimento delle colonne `e_type_*`
            # avg("total_execution_time").alias("avg_total_execution_time"),  # Media del tempo di esecuzione
            # avg("scheduling_time").alias("avg_scheduling_time")  # Media del tempo di scheduling
        )
    )

    # Grouping and calculations
    result_300 = (
        df.groupBy(time_window_300)
        .agg(
            count("*").alias("n_invocations"),  # Numero di invocazioni
            avg("avg_loc").alias("avg_loc"),  # Media delle linee di codice
            avg("avg_cyc_complexity").alias("avg_cyc_complexity"),  # Media della complessità ciclica
            avg("avg_num_of_imports").alias("avg_num_of_imports"),  # Media del numero di importazioni
            avg("avg_argument_size").alias("avg_argument_size"),  # Media della dimensione degli argomenti
            *[first(col(c)).alias(c) for c in endpoint_type_columns],  # Mantenimento delle colonne `e_type_*`
            # avg("total_execution_time").alias("avg_total_execution_time"),  # Media del tempo di esecuzione
            # avg("scheduling_time").alias("avg_scheduling_time")  # Media del tempo di scheduling
        )
    )

    # Add a `timestamp` column based on the start of the time window
    result_30 = result_30.withColumn("timestamp", col("window.start"))
    # Drop the window column, as it's no longer needed
    result_30 = result_30.drop("window")
    result_30 = result_30.orderBy("timestamp")

    result_60 = result_60.withColumn("timestamp", col("window.start"))
    # Drop the window column, as it's no longer needed
    result_60 = result_60.drop("window")
    result_60 = result_60.orderBy("timestamp")

    result_300 = result_300.withColumn("timestamp", col("window.start"))
    # Drop the window column, as it's no longer needed
    result_300 = result_300.drop("window")
    result_300 = result_300.orderBy("timestamp")

    result_30.coalesce(1).write.csv(output_csv+"/30", header=True, mode="overwrite")
    result_60.coalesce(1).write.csv(output_csv+"/60", header=True, mode="overwrite")
    result_300.coalesce(1).write.csv(output_csv+"/300", header=True, mode="overwrite")

    # Convert Spark DataFrame to a list of dictionaries
    list_30 = result_30.rdd.map(lambda row: row.asDict()).collect()
    list_60 = result_60.rdd.map(lambda row: row.asDict()).collect()
    list_300 = result_300.rdd.map(lambda row: row.asDict()).collect()

    # Transform the list of dictionaries into a dictionary of lists
    dict_30 = defaultdict(list)
    for row in list_30:
        for key, value in row.items():
            dict_30[key].append(value)

    dict_60 = defaultdict(list)
    for row in list_60:
        for key, value in row.items():
            dict_60[key].append(value)

    dict_300 = defaultdict(list)
    for row in list_300:
        for key, value in row.items():
            dict_300[key].append(value)

    # Convert defaultdict to regular dict for Hugging Face Dataset
    dict_30 = dict(dict_30)
    dict_60 = dict(dict_60)
    dict_300 = dict(dict_300)

    # Step 9: Convert to Hugging Face Dataset
    hf_30_dataset = Dataset.from_dict(dict_30)
    hf_60_dataset = Dataset.from_dict(dict_60)
    hf_300_dataset = Dataset.from_dict(dict_300)

    # Create DatasetDict
    dataset_dict_30 = DatasetDict({
        'train': hf_30_dataset
    })
    dataset_dict_60 = DatasetDict({
        'train': hf_60_dataset
    })
    dataset_dict_300 = DatasetDict({
        'train': hf_300_dataset
    })

    # Push to Hugging Face Hub
    dataset_dict_30.push_to_hub("anastasiafrosted/endpoint3_30")
    dataset_dict_60.push_to_hub("anastasiafrosted/endpoint3_60")
    dataset_dict_300.push_to_hub("anastasiafrosted/endpoint3_300")

    # Stop Spark session
    spark.stop()


if __name__ == "__main__":
    #time_window("data/globus/globus.csv", "data/globus")
    time_window("data/endpoints/endpoint3/e3.csv", "data/endpoints/endpoint3")