import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_unixtime, col, count, window

WINDOW_SIZE = 120  # Window size in seconds

def get_arrivals(input_csv, output_file, rates_file):
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Melt and Filter RDD") \
        .getOrCreate()

    # Read the CSV file
    df = spark.read.csv(input_csv, header=True, inferSchema=True)

    # Drop unwanted columns
    df = df.drop('argument_size', 'loc', 'cyc_complexity', 'num_of_imports')

    # Convert nanosecond timestamp to seconds by dividing by 1e9
    df = df.withColumn("arrival_timestamp", col("arrival_timestamp") / 1e9)

    # Convert the timestamp to a Spark 'timestamp' type
    df = df.withColumn("arrival_timestamp", from_unixtime(col("arrival_timestamp")).cast("timestamp"))

    # Use window function to group by T-second intervals
    time_window_10 = window(col("arrival_timestamp"), str(WINDOW_SIZE)+" seconds")

    result_10 = (
        df.groupBy(time_window_10)
        .agg(
            count("*").alias("n_arrivals"),  # Count number of arrivals per window
        )
    )

    # Multiply the 'n_arrivals' by 1000 (as requested)
    result_10 = result_10.withColumn("n_arrivals", col("n_arrivals")*10 + 100)

    # Add a `timestamp` column based on the start of the time window
    result_10 = result_10.withColumn("timestamp", col("window.start"))

    # Drop the window column, as it's no longer needed
    result_10 = result_10.drop("window")

    # Order by timestamp
    result_10 = result_10.orderBy("timestamp")

    # Convert the result to Pandas DataFrame for further processing (for numpy usage)
    result_10_pd = result_10.toPandas()

    # Definiamo il numero di righe per ogni finestra (12 righe da 10s = 120s)
    WINDOW_SIZE_ROWS = 12
    WINDOW_DURATION_SEC = 120

    # Raggruppiamo ogni 12 righe e sommiamo gli arrivi
    arrival_rates = result_10_pd["n_arrivals"].groupby(result_10_pd.index // WINDOW_SIZE_ROWS).sum()
    # Calcoliamo il rate dividendo per 120 secondi
    arrival_rates = arrival_rates / WINDOW_DURATION_SEC
    # Convertiamo in DataFrame
    arrival_rates_df = pd.DataFrame({"arrival_rate": arrival_rates.reset_index(drop=True)})
    # Salviamo il DataFrame
    arrival_rates_df.to_csv(rates_file, index=False, header=False)

    # Now let's create the arrival times based on `n_arrivals`
    total_arrivals = int(result_10_pd['n_arrivals'].sum())  # Total number of arrivals
    arrival_times = np.zeros(total_arrivals)  # Initialize the array for arrival times
    arrival_count = 0  # Renamed from 'count' to 'arrival_count'
    rng = np.random.default_rng(123)  # Random number generator

    for _, row in result_10_pd.iterrows():
        t0 = row['timestamp'].timestamp()  # Convert timestamp to seconds
        t1 = t0 + WINDOW_SIZE  # End of the time window
        num_arrivals = int(row['n_arrivals'])  # Number of arrivals in this window

        # Generate random arrival times uniformly distributed within [t0, t1]
        arrival_times[arrival_count:arrival_count + num_arrivals] = np.sort(rng.uniform(t0, t1, num_arrivals))
        arrival_count += num_arrivals  # Increment the arrival count

    # Calculate inter-arrival times
    inter_arrival_times = np.diff(arrival_times)

    # Save inter-arrival times to CSV (no header)
    pd.DataFrame(inter_arrival_times).to_csv(output_file, header=False, index=False)

    # Stop Spark session
    spark.stop()


if __name__ == "__main__":
    for i in range(0,2):
        print(f"working on endpoint: {i}")
        get_arrivals("data/traces/endpoint"+str(i)+"/e"+str(i)+".csv",
                     "data/traces/endpoint"+str(i)+"/synthetic/inter_arrivals"+str(i)+".csv",
                     "data/traces/endpoint"+str(i)+"/synthetic/arrival_rates_120s.csv")
