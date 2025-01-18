import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def generate_interarrival_times(data):

    # Step 1: Load the data into a DataFrame
    df = pd.DataFrame(data)

    # Step 2: Select features for clustering (exclude 'arrival_timestamp')
    arrivals = df['arrival_timestamp'].copy()

    # Step 3: Get first arrival:
    interarrival_times = []
    previous_arrival = arrivals.iloc[0]

    # Step 3: Create interarrivals
    for actual_arrival in arrivals:
        interarrival_times.append((actual_arrival - previous_arrival)/ 1e9)
        previous_arrival = actual_arrival

    pd.Series(interarrival_times).to_csv('data/traces/endpoint0/functions/arrivals1.csv', index=False)

if __name__ == "__main__":
    data = pd.read_csv("data/traces/endpoint0/functions/function1.csv")
    generate_interarrival_times(data)
