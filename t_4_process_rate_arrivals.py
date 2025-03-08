import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def generate_rates(data, window_size=120):
    # Step 1: Caricare i tempi di interarrivo
    inter_arrivals = data.iloc[:, 0].values  # Supponiamo che sia una colonna unica

    # Step 2: Calcolare gli istanti di arrivo
    arrival_times = inter_arrivals.cumsum()

    # Step 3: Creare finestre di 120 secondi e contare gli eventi in ciascuna
    max_time = arrival_times[-1]
    bins = range(0, int(max_time) + window_size, window_size)
    counts, _ = pd.cut(arrival_times, bins, right=False, retbins=True)

    # Step 4: Calcolare il tasso di arrivo (eventi per secondo)
    rates = counts.value_counts().sort_index() / window_size
    pd.Series(rates).to_csv('data/traces/endpoint0/globus0_rates.csv', header=False, index=False)

if __name__ == "__main__":
    data = pd.read_csv("data/traces/endpoint0/inter_arrivals0.csv")
    generate_rates(data)
