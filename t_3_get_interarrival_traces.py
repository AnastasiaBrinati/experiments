import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def generate_interarrival_times(data):

    # Step 1: Load the data_globus into a DataFrame
    df = pd.DataFrame(data)

    # Step 2:
    arrivals = df['arrival_timestamp'].copy()

    # Get the test set, from: 2023-07-02
    # cutoff_timestamp_ns = 1688256000000000000
    # Cut the arrivals
    # arrivals = arrivals[arrivals > cutoff_timestamp_ns]

    # Step 3: Get first arrival
    interarrival_times = []
    previous_arrival = arrivals.iloc[0]

    # Step 3: Create interarrivals
    for actual_arrival in arrivals:
        interarrival_times.append(((actual_arrival - previous_arrival)/ 1e9) / 100)
        previous_arrival = actual_arrival

    pd.Series(interarrival_times).to_csv('data_globus/traces/endpoint0/inter_arrivals0.csv', header=False, index=False)

if __name__ == "__main__":
    data = pd.read_csv("data_globus/traces/endpoint0/e0.csv")
    generate_interarrival_times(data)
