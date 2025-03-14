from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SEED = 123
STEP_LEN = 120  # secondi


def graph(interarrivals, rates, file_path):
    """ Funzione per salvare l'immagine del grafico. """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(interarrivals[:4000])
    axs[0].set_xlabel("Interarrival Time (minutes)")
    axs[0].set_ylabel("Frequency")
    axs[0].set_title("Interarrival Times")
    axs[0].grid(True)

    axs[1].plot(np.arange(len(rates[:2000])) * STEP_LEN, rates[:2000], label=f"Arrival Rate (per {STEP_LEN}s)", marker="o", color="r")
    axs[1].set_xlabel("Time (minutes)")
    axs[1].set_ylabel("Arrival Rate")
    axs[1].set_title("Arrival Rate Over Time")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(file_path)


def main(input_file, rates_output, interarrivals_output):
    spark = SparkSession.builder.appName("ArrivalProcessing").getOrCreate()
    df = spark.read.option("header", "true").csv(input_file)
    df = df.withColumn("Arrivals", col(df.columns[0]).cast("double") * 10)
    df = df.withColumn("Rates", col("Arrivals") / STEP_LEN)

    rates_pd = df.select("Rates").toPandas()
    rates_pd.to_csv(rates_output, index=False, header=True)

    nArrivals = df.select("Arrivals").rdd.map(lambda row: row["Arrivals"]).collect()
    total_arrivals = int(sum(nArrivals))
    arrival_times = np.zeros(total_arrivals)
    count = 0
    rng = np.random.default_rng(SEED)

    print("Starting on times.")
    for i, arrivals in enumerate(nArrivals):
        t0 = STEP_LEN * i
        t1 = t0 + STEP_LEN
        arrival_times[count:count + int(arrivals)] = np.sort(rng.uniform(t0, t1, int(arrivals)))
        count += int(arrivals)

    inter_arrival_times = np.diff(arrival_times)
    inter_arrival_pd = pd.DataFrame(inter_arrival_times, columns=["Interarrival"])
    inter_arrival_pd.to_csv(interarrivals_output, index=False, header=True)

    spark.stop()
    print(f"Generated {len(nArrivals)} rates.")


if __name__ == "__main__":
    input_file = "data/trace/debs15_2.csv"
    rates_output = "../faas-offloading-sim/models/training/debs15_rates_2.csv"
    interarrivals_output = "../faas-offloading-sim/traces/synthetic/debs15_interarrivals_2.csv"
    main(input_file, rates_output, interarrivals_output)

    """
    rates = "data/trace/debs15_rates.csv"
    rates = spark.read.option("header", "true").csv(rates)
    inter_arrival_times = "data/trace/debs15_interarrivals.csv"
    inter_arrival_times = spark.read.option("header", "true").csv(inter_arrival_times)
    graph(inter_arrival_times, rates, "data/trace/img/debs15.png")
    """
