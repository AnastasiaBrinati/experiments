from pyspark.sql import SparkSession
from pyspark.sql.functions import count,unix_timestamp, col, lit, from_unixtime, lead, first, last, date_format, udf, avg, desc, window, expr, min, max
from collections import defaultdict
from datasets import Dataset, DatasetDict
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.types import NumericType, ArrayType, DoubleType
from pyspark.ml import Pipeline  # Import Pipeline
from pyspark.sql.window import Window


def scale(df):
    # Identify numeric columns
    numeric_cols = ["n_invocations","avg_loc","avg_cyc_complexity","avg_num_of_imports","avg_argument_size"]

    # Assemble numeric columns into a feature vector
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")

    # Scale the features
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

    # Create a Pipeline that includes both steps
    pipeline = Pipeline(stages=[assembler, scaler])

    # Fit the pipeline to the data_globus
    model = pipeline.fit(df)

    # Transform the data_globus using the model
    scaled_df = model.transform(df)

    # Unpack the scaled features vector to individual columns
    unpack_vector_udf = udf(lambda vector: vector.toArray().tolist(), ArrayType(DoubleType()))
    scaled_df = scaled_df.withColumn("scaled", unpack_vector_udf("scaled_features"))

    # Assign unpacked values back to respective columns
    for i, col_name in enumerate(numeric_cols):
        scaled_df = scaled_df.withColumn(col_name, col("scaled").getItem(i))

    # Drop the original vector columns
    scaled_df = scaled_df.drop("features", "scaled_features", "scaled")

    # Save the scaled data_globus to a new CSV file
    #output_path = "data_globus/rescaled"
    #scaled_df.coalesce(1).write.csv(output_path, header=True, mode="overwrite")

    # Save the scaler model to a directory
    scaler_model_path = "helpers/scaler"
    model.stages[1].write().overwrite().save(scaler_model_path)

    return scaled_df


def time_window(input_csv, data_name):
    # input_csv: file csv to group

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Melt and Filter RDD") \
        .getOrCreate()

    df = spark.read.csv(input_csv, header=True, inferSchema=True)

    # Ensure the timestamp column is in the correct format
    df = df.withColumn("arrival_timestamp", col("arrival_timestamp").cast("timestamp"))

    # Use window function to group by T-second intervals
    time_window_60 = window(col("arrival_timestamp").cast("timestamp"), f"{60} seconds")
    time_window_120 =  window(col("arrival_timestamp").cast("timestamp"), f"{120} seconds")
    time_window_300 = window(col("arrival_timestamp").cast("timestamp"), f"{300} seconds")
    time_window_3600 = window(col("arrival_timestamp").cast("timestamp"), f"{3600} seconds")
    time_window_86400 = window(col("arrival_timestamp").cast("timestamp"), f"{86400} seconds")

    endpoint_type_columns = [c for c in df.columns if c.startswith("e_type_")]

    # Grouping and calculations
    result_60 = (
        df.groupBy(time_window_60)
        .agg(
            count("*").alias("n_invocations"),  # Numero di invocazioni
            avg("avg_loc").alias("avg_loc"),  # Media delle linee di codice
            avg("avg_cyc_complexity").alias("avg_cyc_complexity"),  # Media della complessità ciclica
            avg("avg_num_of_imports").alias("avg_num_of_imports"),  # Media del numero di importazioni
            avg("avg_argument_size").alias("avg_argument_size"),  # Media della dimensione degli argomenti
        )
        .withColumn(
            "avg_invocations_rate",
            col("n_invocations") / 60
        )
    )

    result_120 = (
        df.groupBy(time_window_120)
        .agg(
            count("*").alias("n_invocations"),
            avg("avg_loc").alias("avg_loc"),
            avg("avg_cyc_complexity").alias("avg_cyc_complexity"),
            avg("avg_num_of_imports").alias("avg_num_of_imports"),
            avg("avg_argument_size").alias("avg_argument_size"),
        )
        .withColumn(
            "avg_invocations_rate",
            col("n_invocations") / 120
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
        )
        .withColumn(
            "avg_invocations_rate",
            col("n_invocations") / 300
        )
    )

    result_3600 = (
        df.groupBy(time_window_3600)
        .agg(
            count("*").alias("n_invocations"),  # Numero di invocazioni
            avg("avg_loc").alias("avg_loc"),  # Media delle linee di codice
            avg("avg_cyc_complexity").alias("avg_cyc_complexity"),  # Media della complessità ciclica
            avg("avg_num_of_imports").alias("avg_num_of_imports"),  # Media del numero di importazioni
            avg("avg_argument_size").alias("avg_argument_size"),  # Media della dimensione degli argomenti
        )
        .withColumn(
            "avg_invocations_rate",
            col("n_invocations") / 3600
        )
    )

    # Grouping and calculations
    result_86400 = (
        df.groupBy(time_window_86400)
        .agg(
            count("*").alias("n_invocations"),  # Numero di invocazioni
            avg("avg_loc").alias("avg_loc"),  # Media delle linee di codice
            avg("avg_cyc_complexity").alias("avg_cyc_complexity"),  # Media della complessità ciclica
            avg("avg_num_of_imports").alias("avg_num_of_imports"),  # Media del numero di importazioni
            avg("avg_argument_size").alias("avg_argument_size"),  # Media della dimensione degli argomenti
        )
        .withColumn(
            "avg_invocations_rate",
            col("n_invocations") / 86400
        )
    )

    # Add a `timestamp` column based on the start of the time window

    result_60 = result_60.withColumn("timestamp", col("window.start"))
    # Drop the window column, as it's no longer needed
    result_60 = result_60.drop("window")
    result_60 = result_60.orderBy("timestamp")

    result_120 = result_120.withColumn("timestamp", col("window.start"))
    # Drop the window column, as it's no longer needed
    result_120 = result_120.drop("window")
    result_120 = result_120.orderBy("timestamp")

    result_300 = result_300.withColumn("timestamp", col("window.start"))
    # Drop the window column, as it's no longer needed
    result_300 = result_300.drop("window")
    result_300 = result_300.orderBy("timestamp")

    result_3600 = result_3600.withColumn("timestamp", col("window.start"))
    # Drop the window column, as it's no longer needed
    result_3600 = result_3600.drop("window")
    result_3600 = result_3600.orderBy("timestamp")

    result_86400 = result_86400.withColumn("timestamp", col("window.start"))
    # Drop the window column, as it's no longer needed
    result_86400 = result_86400.drop("window")
    result_86400 = result_86400.orderBy("timestamp")

    '''
    # scaling
    result_30 = scale(result_30)
    result_60 = scale(result_60)
    result_300 = scale(result_300)

    result_30.coalesce(1).write.csv(output_csv+"/30", header=True, mode="overwrite")
    result_60.coalesce(1).write.csv(output_csv+"/60", header=True, mode="overwrite")
    result_300.coalesce(1).write.csv(output_csv+"/300", header=True, mode="overwrite")
    
    '''

    # Convert Spark DataFrame to a list of dictionaries
    list_60 = result_60.rdd.map(lambda row: row.asDict()).collect()
    list_120 = result_120.rdd.map(lambda row: row.asDict()).collect()
    list_300 = result_300.rdd.map(lambda row: row.asDict()).collect()
    list_3600 = result_3600.rdd.map(lambda row: row.asDict()).collect()
    list_86400 = result_86400.rdd.map(lambda row: row.asDict()).collect()

    # Transform the list of dictionaries into a dictionary of lists
    dict_60 = defaultdict(list)
    for row in list_60:
        for key, value in row.items():
            dict_60[key].append(value)

    dict_120 = defaultdict(list)
    for row in list_120:
        for key, value in row.items():
            dict_120[key].append(value)

    dict_300 = defaultdict(list)
    for row in list_300:
        for key, value in row.items():
            dict_300[key].append(value)

    dict_3600 = defaultdict(list)
    for row in list_3600:
        for key, value in row.items():
            dict_3600[key].append(value)

    dict_86400 = defaultdict(list)
    for row in list_86400:
        for key, value in row.items():
            dict_86400[key].append(value)

    # Convert defaultdict to regular dict for Hugging Face Dataset
    dict_60 = dict(dict_60)
    dict_120 = dict(dict_120)
    dict_300 = dict(dict_300)
    dict_3600 = dict(dict_3600)
    dict_86400 = dict(dict_86400)

    # Step 9: Convert to Hugging Face Dataset
    hf_60_dataset = Dataset.from_dict(dict_60)
    hf_120_dataset = Dataset.from_dict(dict_120)
    hf_300_dataset = Dataset.from_dict(dict_300)
    hf_3600_dataset = Dataset.from_dict(dict_3600)
    hf_86400_dataset = Dataset.from_dict(dict_86400)

    # Create DatasetDict
    dataset_dict_60 = DatasetDict({
        'train': hf_60_dataset
    })
    dataset_dict_120 = DatasetDict({
        'train': hf_120_dataset
    })
    dataset_dict_300 = DatasetDict({
        'train': hf_300_dataset
    })
    dataset_dict_3600 = DatasetDict({
        'train': hf_3600_dataset
    })
    dataset_dict_86400 = DatasetDict({
        'train': hf_86400_dataset
    })

    # Push to Hugging Face Hub
    #dataset_dict_60.push_to_hub("anastasiafrosted/"+data_name+"_60")
    dataset_dict_120.push_to_hub("anastasiafrosted/"+data_name+"_120")
    #dataset_dict_300.push_to_hub("anastasiafrosted/"+data_name+"_300")
    #dataset_dict_3600.push_to_hub("anastasiafrosted/"+data_name+"_3600")
    #dataset_dict_86400.push_to_hub("anastasiafrosted/"+data_name+"_86400")

    # Stop Spark session
    spark.stop()


if __name__ == "__main__":
    #time_window("data_globus/globus/globus.csv", "data_globus/globus")
    #time_window("data_globus/globus/globus.csv", "data_globus/globus/")
    time_window("data_globus/endpoints/endpoint1/e1.csv", "endpoint1")